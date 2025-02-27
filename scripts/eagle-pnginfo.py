import os
import gradio as gr
import re
from datetime import datetime
import logging
from typing import Tuple, List, Optional

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

# 定数
EAGLE_SERVER_URL = "http://localhost"
EAGLE_PORT = 41595
STABLE_DIFFUSION_FOLDER_NAME = "stable diffusion"
MOUNTED_DRIVE_FOLDER = "/content/gdrive/MyDrive/Eagle"
PATH_ROOT = paths.script_path

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)

# -----------------------------------------------------------------------------
# ユーティリティ関数
# -----------------------------------------------------------------------------
def split_prompt(prompt: str) -> List[str]:
    """プロンプトをトークンに分割します。

    Args:
        prompt: 分割するプロンプト文字列

    Returns:
        分割されたトークンのリスト
    """
    tokens = re.split(r",|\s*(?i:break)\s*", prompt)
    return [token.strip() for token in tokens if token.strip()]


def process_prompt(prompt: str, prefix: str = "") -> List[str]:
    """プロンプトを処理し、接頭辞を付けてリストを返します。

    Args:
        prompt: 処理するプロンプト文字列
        prefix: トークンに付ける接頭辞 (デフォルトは空)

    Returns:
        処理されたトークンのリスト
    """
    tokens = split_prompt(prompt)
    return [f"{prefix}{token}" for token in tokens] if prefix else tokens


# -----------------------------------------------------------------------------
# メタデータ抽出と生成
# -----------------------------------------------------------------------------
def extract_prompt_info(
    params: script_callbacks.ImageSaveParams,
) -> Tuple[Optional[str], str, str]:
    """プロンプト情報を抽出します。

    Args:
        params: 画像保存パラメータ

    Returns:
        info, positive_prompt, negative_prompt のタプル
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


def generate_tags(
    params: script_callbacks.ImageSaveParams, positive_prompt: str, negative_prompt: str
) -> Tuple[Optional[str], List[str]]:
    """タグを生成します。

    Args:
        params: 画像保存パラメータ
        positive_prompt: 正のプロンプト
        negative_prompt: 負のプロンプト

    Returns:
        annotation, tags のタプル
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


def create_png_metadata(
    annotation: Optional[str],
    tags: List[str],
    info: Optional[str],
    params: script_callbacks.ImageSaveParams,
) -> PngImagePlugin.PngInfo:
    """PNGメタデータを作成します。

    Args:
        annotation: アノテーション文字列
        tags: タグのリスト
        info: パラメータ情報
        params: 画像保存パラメータ

    Returns:
        PNGメタデータオブジェクト
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
# Eagle用: stable diffusionフォルダの取得または作成
# -----------------------------------------------------------------------------
def fetch_or_create_stable_diffusion_folder(
    server_url: str = EAGLE_SERVER_URL, port: int = EAGLE_PORT
) -> str:
    """stable diffusionフォルダを取得または作成します。

    Args:
        server_url: EagleサーバーのURL
        port: Eagleサーバーのポート

    Returns:
        フォルダID
    """
    logging.debug(f"Fetching or creating '{STABLE_DIFFUSION_FOLDER_NAME}' folder")
    resp = api_folder.list(server_url=server_url, port=port)
    if resp.status_code != 200:
        logging.error(f"Eagleフォルダ一覧取得失敗: {resp.text}")
        return ""
    folder_list = api_util.getAllFolder(resp)
    if not folder_list:
        logging.error("フォルダ一覧が空または解析失敗")
        return ""
    for fd in folder_list:
        if fd.get("name") == STABLE_DIFFUSION_FOLDER_NAME:
            logging.info(
                f"Found existing '{STABLE_DIFFUSION_FOLDER_NAME}' folder: ID={fd.get('id')}"
            )
            return fd.get("id")
    logging.info(f"'{STABLE_DIFFUSION_FOLDER_NAME}' フォルダが無いので新規作成します")
    r_create = api_folder.create(
        STABLE_DIFFUSION_FOLDER_NAME, server_url=server_url, port=port
    )
    if r_create.status_code == 200:
        try:
            new_id = r_create.json()["data"]["id"]
            logging.info(
                f"Created '{STABLE_DIFFUSION_FOLDER_NAME}' folder: ID={new_id}"
            )
            return new_id
        except:
            logging.error("フォルダ作成後のレスポンス解析に失敗")
            return ""
    else:
        logging.error(f"'{STABLE_DIFFUSION_FOLDER_NAME}' フォルダ作成失敗: {r_create.text}")
        return ""


# -----------------------------------------------------------------------------
# Eagle用: 日付サブフォルダの検索または作成
# -----------------------------------------------------------------------------
def find_or_create_subfolder(
    parent_id: str,
    subfolder_name: str,
    server_url: str = EAGLE_SERVER_URL,
    port: int = EAGLE_PORT,
) -> str:
    """日付サブフォルダを検索または作成します。

    Args:
        parent_id: 親フォルダのID
        subfolder_name: サブフォルダ名
        server_url: EagleサーバーのURL
        port: Eagleサーバーのポート

    Returns:
        サブフォルダID
    """
    logging.debug(
        f"find_or_create_subfolder 開始: parent_id={parent_id}, subfolder_name={subfolder_name}"
    )
    r_list = api_folder.list(server_url=server_url, port=port)
    if r_list.status_code != 200:
        logging.error(f"サブフォルダ検索: フォルダ一覧取得失敗: {r_list.text}")
        return ""
    folder_list = api_util.getAllFolder(r_list)
    if not folder_list:
        logging.error("サブフォルダ検索: フォルダ一覧が空または解析失敗")
        return ""
    for fd in folder_list:
        if fd.get("name") == subfolder_name and "stable diffusion" in fd.get(
            "extendTags", []
        ):
            logging.info(f"既存サブフォルダあり: '{subfolder_name}' (ID={fd.get('id')})")
            return fd.get("id")
    logging.info(f"サブフォルダ '{subfolder_name}' が無いので新規作成")
    r_sub = api_folder.create_subfolder(
        newfoldername=subfolder_name,
        parent_id=parent_id,
        server_url=server_url,
        port=port,
        allow_duplicate_name=False,
    )
    if r_sub.status_code == 200:
        try:
            new_id = r_sub.json()["data"]["id"]
            logging.info(f"サブフォルダ '{subfolder_name}' 作成完了: ID={new_id}")
            return new_id
        except Exception as e:
            logging.error(f"サブフォルダ作成レスポンス解析失敗: {str(e)}")
            return ""
    else:
        logging.error(f"サブフォルダ作成失敗: {r_sub.text}")
        return ""


# -----------------------------------------------------------------------------
# 画像保存処理
# -----------------------------------------------------------------------------
def save_image_to_drive(
    image: Image.Image,
    png_metadata: PngImagePlugin.PngInfo,
    filename: str,
    params: script_callbacks.ImageSaveParams,
    main_folder_id: str = "1NuzFVjymjx5ByHPVqYKTDjDj6R3BlKvU",
) -> None:
    """Google Driveに画像を保存します。

    Args:
        image: 保存する画像オブジェクト
        png_metadata: PNGメタデータ
        filename: ファイル名
        params: 画像保存パラメータ
        main_folder_id: メインフォルダID
    """
    date_str = datetime.now().strftime("%Y-%m-%d")
    temp_image_path = os.path.join("/tmp", "temp_" + filename)
    try:
        image.save(temp_image_path, pnginfo=png_metadata)
        logging.info(f"一時画像ファイルを保存しました: {temp_image_path}")
    except Exception as e:
        logging.error("一時画像ファイルの保存に失敗しました")
        logging.error(str(e))
        return
    try:
        credentials = service_account.Credentials.from_service_account_file(
            os.path.join(PATH_ROOT, "service_account.json"),
            scopes=["https://www.googleapis.com/auth/drive"],
        )
        drive_service = build("drive", "v3", credentials=credentials)
        query = f"name='{date_str}' and mimeType='application/vnd.google-apps.folder' and '{main_folder_id}' in parents and trashed=false"
        response = (
            drive_service.files()
            .list(q=query, spaces="drive", fields="files(id, name)")
            .execute()
        )
        files = response.get("files", [])
        if files:
            date_folder_id = files[0]["id"]
            logging.info(f"既存の日付フォルダが見つかりました: {date_str}")
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
            logging.info(f"日付フォルダを作成しました: {date_str}")
        file_metadata = {"name": filename, "parents": [date_folder_id]}
        media = MediaFileUpload(temp_image_path, mimetype="image/png")
        file = (
            drive_service.files()
            .create(body=file_metadata, media_body=media, fields="id")
            .execute()
        )
        logging.info(f"Google Driveにアップロード完了 (ID): {file.get('id')}")
    except Exception as e:
        logging.error("Google Driveへのアップロードに失敗しました")
        logging.error(str(e))
    finally:
        try:
            os.remove(temp_image_path)
            logging.info(f"一時画像ファイルを削除しました: {temp_image_path}")
        except Exception as e:
            logging.error("一時画像ファイルの削除に失敗しました")
            logging.error(str(e))


def save_image_to_mounted_drive(
    image: Image.Image, png_metadata: PngImagePlugin.PngInfo, filename: str
) -> None:
    """マウント済みDriveに画像を保存します。

    Args:
        image: 保存する画像オブジェクト
        png_metadata: PNGメタデータ
        filename: ファイル名
    """
    date_str = datetime.now().strftime("%Y-%m-%d")
    drive_date_folder = os.path.join(MOUNTED_DRIVE_FOLDER, date_str)
    if not os.path.exists(drive_date_folder):
        os.makedirs(drive_date_folder, exist_ok=True)
        logging.info(f"日付フォルダを作成しました: {drive_date_folder}")
    destination_path = os.path.join(drive_date_folder, filename)
    try:
        image.save(destination_path, pnginfo=png_metadata)
        logging.info(f"Colabのマウント済みDriveに保存しました: {destination_path}")
    except Exception as e:
        logging.error("Colabのマウント済みDriveへの保存に失敗しました")
        logging.error(str(e))


def send_image_to_eagle(
    fullfn: str,
    filename: str,
    annotation: Optional[str],
    tags: List[str],
    server_url: str = EAGLE_SERVER_URL,
    port: int = EAGLE_PORT,
) -> None:
    """Eagle APIを使用して画像を送信します。

    Args:
        fullfn: 画像のフルパス
        filename: ファイル名
        annotation: アノテーション
        tags: タグのリスト
        server_url: EagleサーバーのURL
        port: Eagleサーバーのポート
    """
    if not shared.opts.use_local_env:
        logging.info("ローカル環境でEagle転送が無効です")
        return
    logging.info("ローカル環境でEagle転送を試みます")
    stable_folder_id = fetch_or_create_stable_diffusion_folder(
        server_url=server_url, port=port
    )
    if not stable_folder_id:
        logging.error("stable diffusionフォルダの取得または作成に失敗")
        return
    date_str = datetime.now().strftime("%Y-%m-%d")
    target_folder_id = find_or_create_subfolder(
        stable_folder_id, date_str, server_url=server_url, port=port
    )
    if not target_folder_id:
        logging.error(f"日付サブフォルダ '{date_str}' の作成に失敗")
        return
    logging.info(f"日付サブフォルダ '{date_str}' を取得しました (ID={target_folder_id})")
    item = api_item.EAGLE_ITEM_PATH(
        filefullpath=fullfn, filename=filename, annotation=annotation, tags=tags
    )
    _ret = api_item.add_from_path(item=item, folderId=target_folder_id)
    if _ret.status_code == 200:
        logging.info(f"Eagle転送成功: {fullfn}")
    else:
        logging.error(f"Eagle転送失敗: {_ret.status_code}, {_ret.content}")


# -----------------------------------------------------------------------------
# 画像保存処理の統合
# -----------------------------------------------------------------------------
def save_or_send_image(
    image: Image.Image,
    png_metadata: PngImagePlugin.PngInfo,
    filename: str,
    params: script_callbacks.ImageSaveParams,
    annotation: Optional[str],
    tags: List[str],
) -> None:
    """画像を保存または送信します。

    Args:
        image: 保存する画像オブジェクト
        png_metadata: PNGメタデータ
        filename: ファイル名
        params: 画像保存パラメータ
        annotation: アノテーション
        tags: タグのリスト
    """
    fullfn = os.path.join(PATH_ROOT, params.filename)
    logging.debug(f"画像保存処理開始: filename={filename}, fullfn={fullfn}")
    if shared.opts.use_paperspace_env:
        save_image_to_drive(image, png_metadata, filename, params)
    elif shared.opts.use_colab_env:
        save_image_to_mounted_drive(image, png_metadata, filename)
    else:
        send_image_to_eagle(fullfn, filename, annotation, tags)


# -----------------------------------------------------------------------------
# on_image_saved コールバック
# -----------------------------------------------------------------------------
def on_image_saved(params: script_callbacks.ImageSaveParams) -> None:
    """画像保存時のコールバック関数。

    Args:
        params: 画像保存パラメータ
    """
    logging.info("画像処理を開始します。")
    image_path = os.path.join(PATH_ROOT, params.filename)
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
def on_ui_settings() -> None:
    """UI設定を登録します。"""
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
