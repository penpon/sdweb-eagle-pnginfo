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

DEBUG = False
google_drive_folder = "/content/gdrive/MyDrive/Eagle"


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
    # 通常、pnginfo["parameters"] は複数行になっており、
    # 「Negative prompt:」という行が現れるまでが正のプロンプトとみなされます。
    info = params.pnginfo.get("parameters", None)
    if info:
        lines = info.split("\n")
        positive_lines = []
        for line in lines:
            # 「Negative prompt:」で始まる行があればそこで終了
            if line.strip().lower().startswith("negative prompt:"):
                break
            positive_lines.append(line)
        # 各行の前後の余分な空白を除去し、カンマで結合して正のプロンプト全体を作成
        final_pos_prompt = ", ".join(
            [l.strip() for l in positive_lines if l.strip() != ""]
        )

        # ※必要に応じて負のプロンプトも同様に抽出できますが、
        # 今回は pnginfo 内の「Negative prompt:」以降の文字列をそのまま使用します。
        # ここでは、既存の params.p.negative_prompt を fallback とします。
        final_neg_prompt = params.p.negative_prompt
    else:
        final_pos_prompt = params.p.prompt
        final_neg_prompt = params.p.negative_prompt

    # 生成情報（annotation）およびタグを生成する
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

    # 転送先の Google Drive のフォルダパス（※Drive のマウントが完了している前提）
    if not os.path.exists(google_drive_folder):
        try:
            os.makedirs(google_drive_folder, exist_ok=True)
            dprint("DEBUG: Google Drive 転送先フォルダを作成しました: " + google_drive_folder)
        except Exception as e:
            dprint("DEBUG: Google Drive 転送先フォルダの作成に失敗しました")
            dprint(e)
            return

    # 画像生成時の日付ディレクトリを作成（例："2025-02-12"）
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

    # 保存先は日付ディレクトリ内とする
    destination_file = os.path.join(date_folder, basename)
    try:
        im = Image.open(fullfn)
        meta = PngImagePlugin.PngInfo()
        if annotation:
            meta.add_text("Annotation", annotation)
        if tags:
            meta.add_text("Tags", ", ".join(tags))
        # 画像を転送先フォルダに、メタデータ付きで保存（上書き保存ではなく新たなファイルとして作成）
        im.save(destination_file, pnginfo=meta)
        dprint("DEBUG: 画像ファイルに情報を埋め込み、Google Drive に保存しました: " + destination_file)
    except Exception as e:
        dprint("DEBUG: 画像ファイルの情報埋め込みおよび保存に失敗しました")
        dprint(e)


# コールバックの登録
script_callbacks.on_image_saved(on_image_saved)
script_callbacks.on_ui_settings(on_ui_settings)
