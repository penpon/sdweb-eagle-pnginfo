import os
import shutil  # （今回のコードでは使わなくなりますが、他の用途で必要な場合は残しておいてください）
import gradio as gr
import re
from datetime import datetime

from modules import paths, script_callbacks, shared
from scripts.parser import Parser
from scripts.tag_generator import TagGenerator

# Pillow のインポート（PNG 画像へのメタデータ埋め込みに使用）
from PIL import Image, PngImagePlugin

# Paperspace Gradient環境でGoogle Drive APIを利用するためのインポート
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
except ImportError:
    pass

# --- ここからパッチ処理 ---
# shared.loaded_hypernetworkが存在しない場合、Noneを設定
if not hasattr(shared, "loaded_hypernetwork"):
    shared.loaded_hypernetwork = None

# KDiffusionSamplerにdefault_eta属性が存在しない場合、0.0を設定（モジュールパスは環境に合わせて調整してください）
try:
    from modules.sd_samplers import KDiffusionSampler

    if not hasattr(KDiffusionSampler, "default_eta"):
        setattr(KDiffusionSampler, "default_eta", 0.0)
except Exception as e:
    # エラー内容をDEBUG出力（DEBUG=Trueの場合のみ表示）
    print("DEBUG: KDiffusionSampler のパッチ適用に失敗: " + str(e))
# --- ここまでパッチ処理 ---

DEBUG = True
google_drive_folder = "/content/gdrive/MyDrive/Eagle"  # Google Colab環境用


def dprint(msg):
    if DEBUG:
        print(msg)


# -----------------------------------------------------------------------------
# ヘルパー関数：プロンプト文字列の分割
# -----------------------------------------------------------------------------
def split_prompt(prompt):
    """
    プロンプト文字列をカンマまたは "BREAK"（大文字・小文字問わず）を区切り文字として分割します。

    例:
      "BREAK
      <lora:wreal_consolidated:0.5>,<lora:add-detail-xl:1>"
    は ["BREAK", "<lora:wreal_consolidated:0.5>", "<lora:add-detail-xl:1>"] に分割されます。
    """
    tokens = re.split(r",|\s*(?i:break)\s*", prompt)
    tokens = [token.strip() for token in tokens if token and token.strip() != ""]
    return tokens


def process_prompt(prompt, prefix=""):
    """
    プロンプト文字列を分割し、必要に応じて各トークンに接頭辞を付与して返します。
    ※通常、Parser.prompt_to_tags() で既に正のプロンプト全体が対象となっている前提です。
    """
    tokens = split_prompt(prompt)
    if prefix:
        tokens = [f"{prefix}{token}" for token in tokens]
    return tokens


# -----------------------------------------------------------------------------
# 現在のスクリプトフォルダのパスを取得
# -----------------------------------------------------------------------------
path_root = paths.script_path


def on_ui_settings():
    shared.opts.add_option(
        "use_google_colab_env",
        shared.OptionInfo(
            False,
            "Google Colab環境",
            section=("google_drive_transfer", "Eagle Pnginfo"),
        ),
    )
    shared.opts.add_option(
        "use_paperspace_gradient_env",
        shared.OptionInfo(
            False,
            "Paperspace Gradient環境",
            section=("google_drive_transfer", "Eagle Pnginfo"),
        ),
    )
    # Google Drive への転送用設定項目を追加します
    shared.opts.add_option(
        "enable_google_drive_transfer",
        shared.OptionInfo(
            False,
            "Google Drive に転送する",
            section=("google_drive_transfer", "Eagle Pnginfo"),
        ),
    )
    shared.opts.add_option(
        "save_generationinfo_to_google_drive",
        shared.OptionInfo(
            False,
            "生成情報を画像に埋め込む",
            section=("google_drive_transfer", "Eagle Pnginfo"),
        ),
    )
    shared.opts.add_option(
        "save_positive_prompt_to_google_drive",
        shared.OptionInfo(
            False,
            "正のプロンプトをタグとして保存",
            section=("google_drive_transfer", "Eagle Pnginfo"),
        ),
    )
    shared.opts.add_option(
        "save_negative_prompt_to_google_drive",
        shared.OptionInfo(
            "n:tag",
            "負のプロンプトをタグとして保存",
            gr.Radio,
            {"choices": ["None", "tag", "n:tag"]},
            section=("google_drive_transfer", "Eagle Pnginfo"),
        ),
    )
    shared.opts.add_option(
        "use_prompt_parser_for_google_drive",
        shared.OptionInfo(
            False,
            "タグ保存時に prompt parser を使用",
            section=("google_drive_transfer", "Eagle Pnginfo"),
        ),
    )
    shared.opts.add_option(
        "additional_tags_for_google_drive",
        shared.OptionInfo(
            "", "追加タグ (カンマ区切り)", section=("google_drive_transfer", "Eagle Pnginfo")
        ),
    )
    # 転送先フォルダは固定 /content/gdrive/MyDrive/Eagle とするため、オプションは不要です


def on_image_saved(params: script_callbacks.ImageSaveParams):
    if not shared.opts.enable_google_drive_transfer:
        dprint("DEBUG:on_image_saved: Google Drive 転送機能は無効です")
        return
    else:
        dprint("DEBUG:on_image_saved: 転送機能有効。Google Drive に画像を転送します。")

    # 保存済み画像ファイルの絶対パスを取得
    fullfn = os.path.join(path_root, params.filename)
    basename = os.path.basename(fullfn)
    filename_without_ext = os.path.splitext(basename)[0]

    # 生成情報（pnginfo["parameters"]）から正のプロンプト全体を抽出する
    info = params.pnginfo.get("parameters", None)
    if info:
        lines = info.split("\n")
        positive_lines = []
        for line in lines:
            if line.strip().lower().startswith("negative prompt:"):
                break
            positive_lines.append(line)
        final_pos_prompt = ", ".join(
            [l.strip() for l in positive_lines if l.strip() != ""]
        )
        final_neg_prompt = params.p.negative_prompt
    else:
        final_pos_prompt = params.p.prompt
        final_neg_prompt = params.p.negative_prompt

    annotation = None
    tags = []
    if shared.opts.save_generationinfo_to_google_drive:
        annotation = info
    if shared.opts.save_positive_prompt_to_google_drive:
        if final_pos_prompt:
            if shared.opts.use_prompt_parser_for_google_drive:
                tags += Parser.prompt_to_tags(final_pos_prompt)
            else:
                tags += process_prompt(final_pos_prompt)
    if shared.opts.save_negative_prompt_to_google_drive == "tag":
        if final_neg_prompt:
            if shared.opts.use_prompt_parser_for_google_drive:
                tags += Parser.prompt_to_tags(final_neg_prompt)
            else:
                tags += process_prompt(final_neg_prompt)
    elif shared.opts.save_negative_prompt_to_google_drive == "n:tag":
        if final_neg_prompt:
            if shared.opts.use_prompt_parser_for_google_drive:
                tags += [f"n:{x}" for x in Parser.prompt_to_tags(final_neg_prompt)]
            else:
                tags += process_prompt(final_neg_prompt, prefix="n:")

    if shared.opts.additional_tags_for_google_drive:
        gen = TagGenerator(p=params.p, image=params.image)
        _tags = gen.generate_from_p(shared.opts.additional_tags_for_google_drive)
        if _tags:
            tags += _tags

    try:
        im = Image.open(fullfn)
    except Exception as e:
        dprint("DEBUG: 画像ファイルのオープンに失敗しました")
        dprint(e)
        return

    meta = PngImagePlugin.PngInfo()
    if annotation:
        meta.add_text("Annotation", annotation)
    if tags:
        meta.add_text("Tags", ", ".join(tags))
    if info:
        meta.add_text("parameters", info)
    else:
        generation_info = (
            f"{params.p.prompt}\n"
            f"Negative prompt: {params.p.negative_prompt}\n"
            f"Steps: {params.p.steps}, "
            f"Sampler: {getattr(params.p, 'sampler_name', 'N/A')}, "
            f"CFG scale: {params.p.cfg_scale}, "
            f"Seed: {params.p.seed}, "
            f"Size: {params.p.width}x{params.p.height}"
        )
        meta.add_text("parameters", generation_info)

    # 環境に応じた保存処理
    if shared.opts.use_paperspace_gradient_env:
        # Paperspace Gradient環境の場合
        main_folder_id = "1NuzFVjymjx5ByHPVqYKTDjDj6R3BlKvU"
        # 一時ファイルパスを /tmp 以下に作成
        temp_filepath = os.path.join("/tmp", "temp_" + basename)
        try:
            im.save(temp_filepath, pnginfo=meta)
            dprint("DEBUG: 画像ファイルを一時ファイルに保存しました: " + temp_filepath)
        except Exception as e:
            dprint("DEBUG: 一時ファイルへの保存に失敗しました")
            dprint(e)
            return

        try:
            # service_account.jsonはカレントディレクトリに存在している前提
            credentials = service_account.Credentials.from_service_account_file(
                os.path.join(path_root, "service_account.json"),
                scopes=["https://www.googleapis.com/auth/drive"],
            )
            drive_service = build("drive", "v3", credentials=credentials)

            # 日付ディレクトリの作成（または取得）
            date_str = datetime.now().strftime("%Y-%m-%d")
            query = f"name='{date_str}' and mimeType='application/vnd.google-apps.folder' and '{main_folder_id}' in parents and trashed=false"
            response = (
                drive_service.files()
                .list(q=query, spaces="drive", fields="files(id, name)")
                .execute()
            )
            files = response.get("files", [])
            if files:
                date_folder_id = files[0]["id"]
                dprint("DEBUG: 既存の日付フォルダが見つかりました: " + date_str)
            else:
                folder_metadata = {
                    "name": date_str,
                    "mimeType": "application/vnd.google-apps.folder",
                    "parents": [main_folder_id],
                }
                folder = (
                    drive_service.files()
                    .create(body=folder_metadata, fields="id")
                    .execute()
                )
                date_folder_id = folder.get("id")
                dprint("DEBUG: 日付フォルダを作成しました: " + date_str)

            # 画像ファイルのアップロード先として日付フォルダを指定
            file_metadata = {"name": basename, "parents": [date_folder_id]}
            media = MediaFileUpload(temp_filepath, mimetype="image/png")
            file = (
                drive_service.files()
                .create(body=file_metadata, media_body=media, fields="id")
                .execute()
            )
            dprint(
                "DEBUG: 画像ファイルをPaperspace GradientのGoogle Drive日付フォルダにアップロードしました: "
                + str(file.get("id"))
            )
        except Exception as e:
            dprint("DEBUG: Google Driveへのアップロードに失敗しました")
            dprint(e)
        finally:
            try:
                os.remove(temp_filepath)
                dprint("DEBUG: 一時ファイルを削除しました: " + temp_filepath)
            except Exception as e:
                dprint("DEBUG: 一時ファイルの削除に失敗しました")
                dprint(e)
    else:
        # Google Colab環境の場合（従来の処理）
        if not os.path.exists(google_drive_folder):
            try:
                os.makedirs(google_drive_folder, exist_ok=True)
                dprint("DEBUG: Google Drive 転送先フォルダを作成しました: " + google_drive_folder)
            except Exception as e:
                dprint("DEBUG: Google Drive 転送先フォルダの作成に失敗しました")
                dprint(e)
                return

        date_str = datetime.now().strftime("%Y-%m-%d")
        date_folder = os.path.join(google_drive_folder, date_str)
        if not os.path.exists(date_folder):
            try:
                os.makedirs(date_folder, exist_ok=True)
                dprint("DEBUG: 日付ディレクトリを作成しました: " + date_folder)
            except Exception as e:
                dprint("DEBUG: 日付ディレクトリの作成に失敗しました")
                dprint(e)
                return

        destination_file = os.path.join(date_folder, basename)
        try:
            im.save(destination_file, pnginfo=meta)
            dprint("DEBUG: 画像ファイルに情報を埋め込み、Google Drive に保存しました: " + destination_file)
        except Exception as e:
            dprint("DEBUG: 画像ファイルの情報埋め込みおよび保存に失敗しました")
            dprint(e)


# コールバックの登録
script_callbacks.on_image_saved(on_image_saved)
script_callbacks.on_ui_settings(on_ui_settings)
