import os
import shutil  # （今回のコードでは使わなくなりますが、他の用途で必要な場合は残しておいてください）
import gradio as gr

from modules import paths, script_callbacks, shared
from scripts.parser import Parser
from scripts.tag_generator import TagGenerator

# Pillow のインポート（PNG 画像へのメタデータ埋め込みに使用）
from PIL import Image, PngImagePlugin

DEBUG = False


def dprint(msg):
    if DEBUG:
        print(msg)


# 現在のスクリプトフォルダのパスを取得
path_root = paths.script_path


def on_ui_settings():
    # Google Drive への転送用設定項目を追加します
    shared.opts.add_option(
        "enable_google_drive_transfer",
        shared.OptionInfo(
            False,
            "Google Drive に転送する",
            section=("google_drive_transfer", "Google Drive 転送"),
        ),
    )
    shared.opts.add_option(
        "save_generationinfo_to_google_drive",
        shared.OptionInfo(
            False,
            "生成情報を画像に埋め込む",
            section=("google_drive_transfer", "Google Drive 転送"),
        ),
    )
    shared.opts.add_option(
        "save_positive_prompt_to_google_drive",
        shared.OptionInfo(
            False,
            "正のプロンプトをタグとして保存",
            section=("google_drive_transfer", "Google Drive 転送"),
        ),
    )
    shared.opts.add_option(
        "save_negative_prompt_to_google_drive",
        shared.OptionInfo(
            "n:tag",
            "負のプロンプトをタグとして保存",
            gr.Radio,
            {"choices": ["None", "tag", "n:tag"]},
            section=("google_drive_transfer", "Google Drive 転送"),
        ),
    )
    shared.opts.add_option(
        "use_prompt_parser_for_google_drive",
        shared.OptionInfo(
            False,
            "タグ保存時に prompt parser を使用",
            section=("google_drive_transfer", "Google Drive 転送"),
        ),
    )
    shared.opts.add_option(
        "additional_tags_for_google_drive",
        shared.OptionInfo(
            "", "追加タグ (カンマ区切り)", section=("google_drive_transfer", "Google Drive 転送")
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
        info = params.pnginfo.get("parameters", None)
        basename = os.path.basename(fullfn)
        filename_without_ext = os.path.splitext(basename)[0]

        # プロンプト情報の取得
        pos_prompt = params.p.prompt
        neg_prompt = params.p.negative_prompt

        # 生成情報（annotation）およびタグを生成する
        annotation = None
        tags = []
        if shared.opts.save_generationinfo_to_google_drive:
            annotation = info
        if shared.opts.save_positive_prompt_to_google_drive:
            if pos_prompt:
                if shared.opts.use_prompt_parser_for_google_drive:
                    tags += Parser.prompt_to_tags(pos_prompt)
                else:
                    tags += [
                        x.strip() for x in pos_prompt.split(",") if x.strip() != ""
                    ]
        if shared.opts.save_negative_prompt_to_google_drive == "tag":
            if neg_prompt:
                if shared.opts.use_prompt_parser_for_google_drive:
                    tags += Parser.prompt_to_tags(neg_prompt)
                else:
                    tags += [
                        x.strip() for x in neg_prompt.split(",") if x.strip() != ""
                    ]
        elif shared.opts.save_negative_prompt_to_google_drive == "n:tag":
            if neg_prompt:
                if shared.opts.use_prompt_parser_for_google_drive:
                    tags += [f"n:{x}" for x in Parser.prompt_to_tags(neg_prompt)]
                else:
                    tags += [
                        f"n:{x.strip()}"
                        for x in neg_prompt.split(",")
                        if x.strip() != ""
                    ]
        if shared.opts.additional_tags_for_google_drive:
            gen = TagGenerator(p=params.p, image=params.image)
            _tags = gen.generate_from_p(shared.opts.additional_tags_for_google_drive)
            if _tags:
                tags += _tags

        # 転送先の Google Drive のフォルダパス（※Drive のマウントが完了している前提）
        google_drive_folder = "/content/gdrive/MyDrive/Eagle"
        if not os.path.exists(google_drive_folder):
            try:
                os.makedirs(google_drive_folder, exist_ok=True)
                dprint("DEBUG: Google Drive 転送先フォルダを作成しました: " + google_drive_folder)
            except Exception as e:
                dprint("DEBUG: Google Drive 転送先フォルダの作成に失敗しました")
                dprint(e)
                return

        # 画像ファイルに追加情報を埋め込む
        destination_file = os.path.join(google_drive_folder, basename)
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
