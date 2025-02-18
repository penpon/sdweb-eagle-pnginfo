import os
import shutil  # ※今回のコードでは使用しないが、他用途用に残しておく
import gradio as gr
import re
from datetime import datetime

from modules import paths, script_callbacks, shared
from scripts.parser import Parser  # prompt_parserは使わないので利用しません
from scripts.tag_generator import TagGenerator

# Pillow: PNG画像へのメタデータ埋め込み用
from PIL import Image, PngImagePlugin

# Paperspace Gradient環境用: Google Drive API
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
except ImportError:
    pass

DEBUG = False
# Google Colab環境でマウント済みDriveのパス
mounted_drive_folder = "/content/gdrive/MyDrive/Eagle"
# スクリプトフォルダのパス
path_root = paths.script_path


def dprint(msg):
    if DEBUG:
        print(msg)


# -----------------------------------------------------------------------------
# プロンプト文字列の処理
# -----------------------------------------------------------------------------
def split_prompt(prompt):
    """
    プロンプト文字列をカンマまたは "BREAK"（大文字・小文字問わず）で分割する。
    例:
      "BREAK\n<lora:wreal_consolidated:0.5>,<lora:add-detail-xl:1>"
    → ["BREAK", "<lora:wreal_consolidated:0.5>", "<lora:add-detail-xl:1>"]
    """
    tokens = re.split(r",|\s*(?i:break)\s*", prompt)
    return [token.strip() for token in tokens if token.strip()]


def process_prompt(prompt, prefix=""):
    """
    分割済みプロンプトに必要なら接頭辞を付与して返す。
    """
    tokens = split_prompt(prompt)
    return [f"{prefix}{token}" for token in tokens] if prefix else tokens


# -----------------------------------------------------------------------------
# プロンプト情報の抽出とタグ生成
# -----------------------------------------------------------------------------
def extract_prompt_info(params):
    """
    pnginfoから正のプロンプト（および負のプロンプトのfallback）を抽出する。
    """
    info = params.pnginfo.get("parameters")
    if info:
        lines = info.split("\n")
        positive_lines = []
        for line in lines:
            if line.strip().lower().startswith("negative prompt:"):
                break
            positive_lines.append(line.strip())
        final_positive = ", ".join([l for l in positive_lines if l])
        final_negative = params.p.negative_prompt
    else:
        final_positive = params.p.prompt
        final_negative = params.p.negative_prompt
    return info, final_positive, final_negative


def generate_tags(params, positive_prompt, negative_prompt):
    """
    オプションに応じてタグおよびAnnotationを生成する。
    タグ生成は常に process_prompt を利用する。
    """
    annotation = (
        params.pnginfo.get("parameters") if shared.opts.embed_generation_info else None
    )
    tags = []
    if shared.opts.save_positive_prompt_tags and positive_prompt:
        tags += process_prompt(positive_prompt)
    if negative_prompt:
        if shared.opts.save_negative_prompt_tags == "tag":
            tags += process_prompt(negative_prompt)
        elif shared.opts.save_negative_prompt_tags == "n:tag":
            tags += process_prompt(negative_prompt, prefix="n:")
    if shared.opts.additional_tags:
        tag_gen = TagGenerator(p=params.p, image=params.image)
        additional = tag_gen.generate_from_p(shared.opts.additional_tags)
        if additional:
            tags += additional
    return annotation, tags


def create_png_metadata(annotation, tags, info, params):
    """
    PNG画像に埋め込むメタデータを生成する。
    """
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
    return meta


# -----------------------------------------------------------------------------
# 画像のGoogle Drive保存処理（環境に応じたアップロード方法を統合）
# -----------------------------------------------------------------------------
def save_image_to_drive(
    image, png_metadata, filename, main_folder_id="1NuzFVjymjx5ByHPVqYKTDjDj6R3BlKvU"
):
    date_str = datetime.now().strftime("%Y-%m-%d")
    if shared.opts.use_paperspace_env:
        # Paperspace環境の場合: Google Drive API を利用してアップロード
        temp_image_path = os.path.join("/tmp", "temp_" + filename)
        try:
            image.save(temp_image_path, pnginfo=png_metadata)
            dprint("DEBUG: 一時画像ファイルを保存しました: " + temp_image_path)
        except Exception as e:
            dprint("DEBUG: 一時画像ファイルの保存に失敗しました")
            dprint(e)
            return

        try:
            credentials = service_account.Credentials.from_service_account_file(
                os.path.join(path_root, "service_account.json"),
                scopes=["https://www.googleapis.com/auth/drive"],
            )
            drive_service = build("drive", "v3", credentials=credentials)
            query = (
                f"name='{date_str}' and mimeType='application/vnd.google-apps.folder' "
                f"and '{main_folder_id}' in parents and trashed=false"
            )
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
            file_metadata = {"name": filename, "parents": [date_folder_id]}
            media = MediaFileUpload(temp_image_path, mimetype="image/png")
            file = (
                drive_service.files()
                .create(body=file_metadata, media_body=media, fields="id")
                .execute()
            )
            dprint("DEBUG: Google Driveにアップロード完了 (ID): " + str(file.get("id")))
        except Exception as e:
            dprint("DEBUG: Google Driveへのアップロードに失敗しました")
            dprint(e)
        finally:
            try:
                os.remove(temp_image_path)
                dprint("DEBUG: 一時画像ファイルを削除しました: " + temp_image_path)
            except Exception as e:
                dprint("DEBUG: 一時画像ファイルの削除に失敗しました")
                dprint(e)
    else:
        # Colab環境の場合: マウント済みDriveに直接保存
        if not os.path.exists(mounted_drive_folder):
            os.makedirs(mounted_drive_folder, exist_ok=True)
            dprint("DEBUG: マウント済みDriveフォルダを作成しました: " + mounted_drive_folder)
        drive_date_folder = os.path.join(mounted_drive_folder, date_str)
        if not os.path.exists(drive_date_folder):
            os.makedirs(drive_date_folder, exist_ok=True)
            dprint("DEBUG: 日付フォルダを作成しました: " + drive_date_folder)
        destination_path = os.path.join(drive_date_folder, filename)
        try:
            image.save(destination_path, pnginfo=png_metadata)
            dprint("DEBUG: Colabのマウント済みDriveに保存しました: " + destination_path)
        except Exception as e:
            dprint("DEBUG: Colabのマウント済みDriveへの保存に失敗しました")
            dprint(e)


# -----------------------------------------------------------------------------
# on_image_saved コールバック
# -----------------------------------------------------------------------------
def on_image_saved(params: script_callbacks.ImageSaveParams):
    if not shared.opts.enable_drive_transfer:
        dprint("DEBUG: Drive 転送機能は無効です")
        return
    dprint("DEBUG: Drive 転送機能有効。画像処理を開始します。")

    image_path = os.path.join(path_root, params.filename)
    filename = os.path.basename(image_path)

    # プロンプト情報の抽出とタグ生成
    info, positive_prompt, negative_prompt = extract_prompt_info(params)
    annotation, tags = generate_tags(params, positive_prompt, negative_prompt)

    try:
        image_obj = Image.open(image_path)
    except Exception as e:
        dprint("DEBUG: 画像ファイルのオープンに失敗しました")
        dprint(e)
        return

    png_metadata = create_png_metadata(annotation, tags, info, params)
    save_image_to_drive(image_obj, png_metadata, filename)


# -----------------------------------------------------------------------------
# UI設定の登録
# -----------------------------------------------------------------------------
def on_ui_settings():
    shared.opts.add_option(
        "use_colab_env",
        shared.OptionInfo(
            False, "Google Colab環境", section=("google_drive_transfer", "Eagle Pnginfo")
        ),
    )
    shared.opts.add_option(
        "use_paperspace_env",
        shared.OptionInfo(
            False,
            "Paperspace Gradient環境",
            section=("google_drive_transfer", "Eagle Pnginfo"),
        ),
    )
    shared.opts.add_option(
        "enable_drive_transfer",
        shared.OptionInfo(
            False,
            "Google Drive に転送する",
            section=("google_drive_transfer", "Eagle Pnginfo"),
        ),
    )
    shared.opts.add_option(
        "embed_generation_info",
        shared.OptionInfo(
            False, "生成情報を画像に埋め込む", section=("google_drive_transfer", "Eagle Pnginfo")
        ),
    )
    shared.opts.add_option(
        "save_positive_prompt_tags",
        shared.OptionInfo(
            False, "正のプロンプトをタグとして保存", section=("google_drive_transfer", "Eagle Pnginfo")
        ),
    )
    shared.opts.add_option(
        "save_negative_prompt_tags",
        shared.OptionInfo(
            "n:tag",
            "負のプロンプトをタグとして保存",
            gr.Radio,
            {"choices": ["None", "tag", "n:tag"]},
            section=("google_drive_transfer", "Eagle Pnginfo"),
        ),
    )
    shared.opts.add_option(
        "additional_tags",
        shared.OptionInfo(
            "", "追加タグ (カンマ区切り)", section=("google_drive_transfer", "Eagle Pnginfo")
        ),
    )


# -----------------------------------------------------------------------------
# コールバックの登録
# -----------------------------------------------------------------------------
script_callbacks.on_image_saved(on_image_saved)
script_callbacks.on_ui_settings(on_ui_settings)
