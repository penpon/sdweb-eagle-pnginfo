import os
import gradio as gr
import re
from datetime import datetime
import logging

from modules import paths, script_callbacks, shared
from scripts.parser import Parser
from scripts.tag_generator import TagGenerator
from scripts.eagleapi import api_application, api_item, api_util, api_folder

from PIL import Image, PngImagePlugin

# Paperspace Gradient環境用: Google Drive API
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
except ImportError:
    pass


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,  # 既存のハンドラを強制的にリセット
)

mounted_drive_folder = "/content/gdrive/MyDrive/Eagle"
path_root = paths.script_path
STABLE_DIFFUSION_NAME = "stable diffusion"

# -----------------------------------------------------------------------------
# プロンプト文字列の処理
# -----------------------------------------------------------------------------
def split_prompt(prompt):
    tokens = re.split(r",|\s*(?i:break)\s*", prompt)
    return [token.strip() for token in tokens if token.strip()]


def process_prompt(prompt, prefix=""):
    tokens = split_prompt(prompt)
    return [f"{prefix}{token}" for token in tokens] if prefix else tokens


# -----------------------------------------------------------------------------
# プロンプト情報の抽出とタグ生成
# -----------------------------------------------------------------------------
def extract_prompt_info(params):
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
# Eagle用: stable diffusionフォルダの取得または作成
# -----------------------------------------------------------------------------
def fetch_or_create_stable_diffusion_folder(server_url="http://localhost", port=41595):
    logging.debug(f"Fetching or creating '{STABLE_DIFFUSION_NAME}' folder")
    resp = api_folder.list(server_url=server_url, port=port)
    if resp.status_code != 200:
        logging.error(f"Eagleフォルダ一覧取得失敗: {resp.text}")
        return ""

    folder_list = api_util.getAllFolder(resp)
    if not folder_list:
        logging.error("フォルダ一覧が空または解析失敗")
        return ""

    for fd in folder_list:
        if fd.get("name") == STABLE_DIFFUSION_NAME:
            logging.info(
                f"Found existing '{STABLE_DIFFUSION_NAME}' folder: ID={fd.get('id')}"
            )
            return fd.get("id")

    logging.info(f"'{STABLE_DIFFUSION_NAME}' フォルダが無いので新規作成します")
    r_create = api_folder.create(
        STABLE_DIFFUSION_NAME, server_url=server_url, port=port
    )
    if r_create.status_code == 200:
        try:
            new_id = r_create.json()["data"]["id"]
            logging.info(f"Created '{STABLE_DIFFUSION_NAME}' folder: ID={new_id}")
            return new_id
        except:
            logging.error("フォルダ作成後のレスポンス解析に失敗")
            return ""
    else:
        logging.error(f"'{STABLE_DIFFUSION_NAME}' フォルダ作成失敗: {r_create.text}")
        return ""


# -----------------------------------------------------------------------------
# Eagle用: 日付サブフォルダの検索または作成（改善版）
# -----------------------------------------------------------------------------
def find_or_create_subfolder(
    parent_id, subfolder_name, server_url="http://localhost", port=41595
):
    logging.debug(
        f"find_or_create_subfolder 開始: parent_id={parent_id}, subfolder_name={subfolder_name}, server_url={server_url}, port={port}"
    )

    # Eagle APIからフォルダ一覧を取得
    logging.debug("フォルダ一覧取得のために api_folder.list を呼び出します")
    r_list = api_folder.list(server_url=server_url, port=port)
    logging.debug(
        f"api_folder.list レスポンス: status_code={r_list.status_code}, text={r_list.text[:100]}..."
    )

    if r_list.status_code != 200:
        logging.error(
            f"サブフォルダ検索: フォルダ一覧取得失敗: status_code={r_list.status_code}, response={r_list.text}"
        )
        return ""

    # フォルダ一覧を解析
    logging.debug("フォルダ一覧を解析するために api_util.getAllFolder を呼び出します")
    folder_list = api_util.getAllFolder(r_list)
    logging.debug(f"取得したフォルダ数: {len(folder_list) if folder_list else 0}")

    if not folder_list:
        logging.error("サブフォルダ検索: フォルダ一覧が空または解析失敗")
        return ""

    # 既存フォルダを検索（parent_idを条件から除外）
    logging.info(f"サබフォルダ '{subfolder_name}' の検索を開始")
    for fd in folder_list:
        fd_name = fd.get("name")
        fd_parent = fd.get("parent")
        fd_tags = fd.get("extendTags", [])
        logging.debug(
            f"チェック中のフォルダ: name={fd_name}, parent={fd_parent}, extendTags={fd_tags}, id={fd.get('id')}"
        )

        if fd_name == subfolder_name and "stable diffusion" in fd_tags:
            logging.info(f"既存サブフォルダあり: '{subfolder_name}' (ID={fd.get('id')})")
            logging.debug(
                f"一致条件: name={fd_name}, 'stable diffusion' in extendTags={fd_tags}"
            )
            return fd.get("id")

    # 既存フォルダが見つからない場合、新規作成
    logging.info(f"サブフォルダ '{subfolder_name}' が無いので新規作成")
    logging.debug(
        f"api_folder.create_subfolder 呼び出し: newfoldername={subfolder_name}, parent_id={parent_id}, allow_duplicate_name=False"
    )
    r_sub = api_folder.create_subfolder(
        newfoldername=subfolder_name,
        parent_id=parent_id,
        server_url=server_url,
        port=port,
        allow_duplicate_name=False,
    )
    logging.debug(
        f"create_subfolder レスポンス: status_code={r_sub.status_code}, text={r_sub.text[:100]}..."
    )

    if r_sub.status_code == 200:
        try:
            new_id = r_sub.json()["data"]["id"]
            logging.info(f"サブフォルダ '{subfolder_name}' 作成完了: ID={new_id}")
            logging.debug(f"レスポンスJSON解析結果: id={new_id}")
            return new_id
        except Exception as e:
            logging.error(
                f"サブフォルダ作成レスポンス解析失敗: exception={str(e)}, response={r_sub.text}"
            )
            return ""
    else:
        logging.error(
            f"サブフォルダ作成失敗: status_code={r_sub.status_code}, response={r_sub.text}"
        )
        return ""


# -----------------------------------------------------------------------------
# 画像保存処理（Google DriveまたはEagle API）
# -----------------------------------------------------------------------------
def save_or_send_image(
    image,
    png_metadata,
    filename,
    params,
    annotation,
    tags,
    main_folder_id="1NuzFVjymjx5ByHPVqYKTDjDj6R3BlKvU",
):
    date_str = datetime.now().strftime("%Y-%m-%d")
    fullfn = os.path.join(path_root, params.filename)
    logging.debug(f"画像保存処理開始: filename={filename}, fullfn={fullfn}")

    # Paperspace環境: Google Drive APIを利用
    if shared.opts.use_paperspace_env:
        temp_image_path = os.path.join("/tmp", "temp_" + filename)
        try:
            image.save(temp_image_path, pnginfo=png_metadata)
            logging.info("一時画像ファイルを保存しました: " + temp_image_path)
        except Exception as e:
            logging.error("一時画像ファイルの保存に失敗しました")
            logging.error(str(e))
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
                logging.info("既存の日付フォルダが見つかりました: " + date_str)
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
                logging.info("日付フォルダを作成しました: " + date_str)
            file_metadata = {"name": filename, "parents": [date_folder_id]}
            media = MediaFileUpload(temp_image_path, mimetype="image/png")
            file = (
                drive_service.files()
                .create(body=file_metadata, media_body=media, fields="id")
                .execute()
            )
            logging.info("Google Driveにアップロード完了 (ID): " + str(file.get("id")))
        except Exception as e:
            logging.error("Google Driveへのアップロードに失敗しました")
            logging.error(str(e))
        finally:
            try:
                os.remove(temp_image_path)
                logging.info("一時画像ファイルを削除しました: " + temp_image_path)
            except Exception as e:
                logging.error("一時画像ファイルの削除に失敗しました")
                logging.error(str(e))
    # Colab環境: マウント済みDriveに直接保存
    elif shared.opts.use_colab_env:
        if not os.path.exists(mounted_drive_folder):
            os.makedirs(mounted_drive_folder, exist_ok=True)
            logging.info("マウント済みDriveフォルダを作成しました: " + mounted_drive_folder)
        drive_date_folder = os.path.join(mounted_drive_folder, date_str)
        if not os.path.exists(drive_date_folder):
            os.makedirs(drive_date_folder, exist_ok=True)
            logging.info("日付フォルダを作成しました: " + drive_date_folder)
        destination_path = os.path.join(drive_date_folder, filename)
        try:
            image.save(destination_path, pnginfo=png_metadata)
            logging.info("Colabのマウント済みDriveに保存しました: " + destination_path)
        except Exception as e:
            logging.error("Colabのマウント済みDriveへの保存に失敗しました")
            logging.error(str(e))
    # ローカル環境: Eagle APIを利用して転送（日付サブフォルダ付き）
    else:
        if not shared.opts.use_local_env:
            logging.info("ローカル環境でEagle転送が無効です")
            return
        logging.info("ローカル環境でEagle転送を試みます")

        server_url = "http://localhost"
        port = 41595

        # stable diffusion フォルダの取得または作成
        stable_folder_id = fetch_or_create_stable_diffusion_folder(
            server_url=server_url, port=port
        )
        if not stable_folder_id:
            logging.error("stable diffusionフォルダの取得または作成に失敗")
            return

        # 日付サブフォルダの取得または作成
        subfolder_name = date_str
        target_folder_id = find_or_create_subfolder(
            stable_folder_id, subfolder_name, server_url=server_url, port=port
        )
        if not target_folder_id:
            logging.error(f"日付サブフォルダ '{subfolder_name}' の作成に失敗")
            return
        else:
            logging.info(f"日付サブフォルダ '{subfolder_name}' を取得しました (ID={target_folder_id})")

        # ローカルEagleに送信
        logging.info("ローカルEagleに送信")
        item = api_item.EAGLE_ITEM_PATH(
            filefullpath=fullfn, filename=filename, annotation=annotation, tags=tags
        )
        _ret = api_item.add_from_path(item=item, folderId=target_folder_id)
        if _ret.status_code == 200:
            logging.info(f"Eagle転送成功: {fullfn}")
        else:
            logging.error(f"Eagle転送失敗: {_ret.status_code}, {_ret.content}")


# -----------------------------------------------------------------------------
# on_image_saved コールバック
# -----------------------------------------------------------------------------
def on_image_saved(params: script_callbacks.ImageSaveParams):
    logging.info("画像処理を開始します。")

    image_path = os.path.join(path_root, params.filename)
    filename = os.path.basename(image_path)
    logging.debug(f"Image path: {image_path}, filename: {filename}")

    info, positive_prompt, negative_prompt = extract_prompt_info(params)
    annotation, tags = generate_tags(params, positive_prompt, negative_prompt)

    try:
        image_obj = Image.open(image_path)
        logging.debug(f"画像ファイルを開きました: {image_path}")
    except Exception as e:
        logging.error("画像ファイルのオープンに失敗しました")
        logging.error(str(e))
        return

    png_metadata = create_png_metadata(annotation, tags, info, params)
    save_or_send_image(image_obj, png_metadata, filename, params, annotation, tags)


# -----------------------------------------------------------------------------
# UI設定の登録
# -----------------------------------------------------------------------------
def on_ui_settings():
    shared.opts.add_option(
        "use_colab_env",
        shared.OptionInfo(
            False, "Google Colab環境", section=("eagle_pnginfo", "Eagle Pnginfo")
        ),
    )
    shared.opts.add_option(
        "use_paperspace_env",
        shared.OptionInfo(
            False, "Paperspace Gradient環境", section=("eagle_pnginfo", "Eagle Pnginfo")
        ),
    )
    shared.opts.add_option(
        "use_local_env",
        shared.OptionInfo(False, "ローカル環境", section=("eagle_pnginfo", "Eagle Pnginfo")),
    )
    shared.opts.add_option(
        "embed_generation_info",
        shared.OptionInfo(
            False, "生成情報を画像に埋め込む", section=("eagle_pnginfo", "Eagle Pnginfo")
        ),
    )
    shared.opts.add_option(
        "save_positive_prompt_tags",
        shared.OptionInfo(
            False, "正のプロンプトをタグとして保存", section=("eagle_pnginfo", "Eagle Pnginfo")
        ),
    )
    shared.opts.add_option(
        "save_negative_prompt_tags",
        shared.OptionInfo(
            "n:tag",
            "負のプロンプトをタグとして保存",
            gr.Radio,
            {"choices": ["None", "tag", "n:tag"]},
            section=("eagle_pnginfo", "Eagle Pnginfo"),
        ),
    )
    shared.opts.add_option(
        "additional_tags",
        shared.OptionInfo(
            "", "追加タグ (カンマ区切り)", section=("eagle_pnginfo", "Eagle Pnginfo")
        ),
    )


# -----------------------------------------------------------------------------
# コールバックの登録
# -----------------------------------------------------------------------------
script_callbacks.on_image_saved(on_image_saved)
script_callbacks.on_ui_settings(on_ui_settings)
