#!/usr/bin/env python3
import os
import sys
import time
import logging
import hashlib
import datetime
import threading
import concurrent.futures

from watchdog.observers.polling import PollingObserver as Observer
from watchdog.events import FileSystemEventHandler
from PIL import Image

# ------------------------------------------------------------------------
# 1) scripts/eagleapi を import できるように sys.path を通す
# ------------------------------------------------------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from scripts.eagleapi import api_folder, api_item, api_util
except ImportError as e:
    logging.error("Eagle API のインポートに失敗。: " + str(e))
    sys.exit(1)

# ------------------------------------------------------------------------
# 定数
# ------------------------------------------------------------------------
EAGLE_SERVER_URL = "http://localhost"
EAGLE_SERVER_PORT = 41595
STABLE_DIFFUSION_NAME = "stable diffusion"

PROCESSED_DB_FILE = os.path.join(os.path.dirname(__file__), "processed_files.txt")
DEFAULT_EAGLE_FOLDER_ID = ""  # サブフォルダ名が取れなかったらルートへ入れる

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# ------------------------------------------------------------------------
# 2) 「stable diffusion」フォルダを探す or 作る
# ------------------------------------------------------------------------
def fetch_or_create_stable_diffusion_folder():
    resp = api_folder.list(server_url=EAGLE_SERVER_URL, port=EAGLE_SERVER_PORT)
    if resp.status_code != 200:
        logging.error("Eagleフォルダ一覧取得失敗:" + resp.text)
        return ""

    folder_list = api_util.getAllFolder(resp)
    if not folder_list:
        logging.error("フォルダ一覧が空または解析失敗。")
        return ""

    for fd in folder_list:
        if fd.get("name") == STABLE_DIFFUSION_NAME:
            return fd.get("id")

    # なければルートに作成
    logging.info(f"'{STABLE_DIFFUSION_NAME}' フォルダが無いので新規作成します。")
    r_create = api_folder.create(
        STABLE_DIFFUSION_NAME, server_url=EAGLE_SERVER_URL, port=EAGLE_SERVER_PORT
    )
    if r_create.status_code == 200:
        try:
            return r_create.json()["data"]["id"]
        except:
            logging.error("フォルダ作成後のレスポンス解析に失敗")
            return ""
    else:
        logging.error(f"'{STABLE_DIFFUSION_NAME}' フォルダ作成失敗: " + r_create.text)
        return ""


# ------------------------------------------------------------------------
# 3) stable diffusion配下に日付サブフォルダを探す or 作る
#    (親ID は stable diffusion フォルダID、重複は extendTags + name でチェック)
# ------------------------------------------------------------------------
def find_or_create_subfolder(parent_id, subfolder_name):
    r_list = api_folder.list(server_url=EAGLE_SERVER_URL, port=EAGLE_SERVER_PORT)
    if r_list.status_code != 200:
        logging.error("サブフォルダ検索: フォルダ一覧取得失敗")
        return ""

    folder_list = api_util.getAllFolder(r_list)
    if not folder_list:
        logging.error("サブフォルダ検索: フォルダ一覧が空または解析失敗")
        return ""

    # --- 「extendTags に 'stable diffusion' があり、name が subfolder_name」のフォルダを探す
    for fd in folder_list:
        if fd.get("name") == subfolder_name and "stable diffusion" in fd.get(
            "extendTags", []
        ):
            logging.info(f"既存サブフォルダあり: '{subfolder_name}' (ID={fd.get('id')})")
            return fd.get("id")

    # 無い場合 => 作成
    logging.info(f"サブフォルダ '{subfolder_name}' が無いので新規作成")
    r_sub = api_folder.create_subfolder(
        newfoldername=subfolder_name,
        parent_id=parent_id,  # stable diffusion フォルダを親にする
        server_url=EAGLE_SERVER_URL,
        port=EAGLE_SERVER_PORT,
        allow_duplicate_name=False,
    )
    if r_sub.status_code == 200:
        try:
            new_id = r_sub.json()["data"]["id"]
            logging.info(f"サブフォルダ'{subfolder_name}'作成完了: ID={new_id}")
            return new_id
        except:
            logging.error("サブフォルダ作成レスポンス解析失敗")
            return ""
    else:
        logging.error(f"サブフォルダ作成失敗: {r_sub.text}")
        return ""


# ------------------------------------------------------------------------
# 4) 重複チェック用
# ------------------------------------------------------------------------
def load_processed_hashes():
    s = set()
    if os.path.exists(PROCESSED_DB_FILE):
        with open(PROCESSED_DB_FILE, "r") as f:
            for line in f:
                h = line.strip()
                if h:
                    s.add(h)
    return s


def save_processed_hash(h):
    with open(PROCESSED_DB_FILE, "a") as f:
        f.write(h + "\n")


# ------------------------------------------------------------------------
# compute_md5: キャッシュなしで毎回 MD5 を計算する
# ------------------------------------------------------------------------
def compute_md5(file_path, block_size=65536):
    logging.info(f"MD5計算開始: {file_path}")
    m = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            while True:
                block = f.read(block_size)
                if not block:
                    break
                m.update(block)
    except Exception as e:
        logging.error(f"ファイル読み込み失敗: {file_path}, err={e}")
        return None
    result = m.hexdigest()
    logging.info(f"MD5計算完了: {file_path} -> {result}")
    return result


def wait_for_file_complete(file_path, timeout=10):
    start = time.time()
    last_size = -1
    while True:
        if time.time() - start > timeout:
            logging.warning("ファイルサイズ安定待ちタイムアウト: " + file_path)
            return False
        try:
            sz = os.path.getsize(file_path)
        except:
            sz = -1
        if sz == last_size and sz != -1:
            return True
        last_size = sz
        time.sleep(0.5)


def get_date_from_file_mtime(file_path):
    ts = os.path.getmtime(file_path)
    return time.strftime("%Y-%m-%d", time.localtime(ts))


# ------------------------------------------------------------------------
# 5) Watchdogハンドラ (並列処理とロックによる重複排除を導入)
# ------------------------------------------------------------------------
class NewFileHandler(FileSystemEventHandler):
    def __init__(self, monitored_folders, stable_folder_id):
        super().__init__()
        self.monitored_folders = monitored_folders
        self.stable_folder_id = stable_folder_id
        self.processed_hashes = load_processed_hashes()
        self.processed_lock = threading.Lock()  # processed_hashes への排他アクセス用
        # 並列処理用スレッドプール（必要に応じて max_workers を調整）
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)

    def process_file(self, file_path):
        # 対象拡張子のみ処理
        if not file_path.lower().endswith((".png", ".jpg", ".jpeg")):
            return

        logging.info(f"ファイル検知: {file_path}")
        if not wait_for_file_complete(file_path):
            logging.info(f"書き込み中っぽいのでスキップ: {file_path}")
            return

        file_hash = compute_md5(file_path)
        if not file_hash:
            return

        # すでに処理済みかチェック（排他制御）
        with self.processed_lock:
            if file_hash in self.processed_hashes:
                logging.info(f"すでに処理済み: {file_path}")
                return
            self.processed_hashes.add(file_hash)

        # 画像読み込み & メタ情報抽出
        try:
            im = Image.open(file_path)
            annotation = im.info.get("Annotation", "")
            tags_str = im.info.get("Tags", "")
            tags = [t.strip() for t in tags_str.split(",") if t.strip()]

            # ファイル更新日時から日付フォルダを決定
            date_dir = get_date_from_file_mtime(file_path)
            logging.info(f"サブフォルダ決定: '{date_dir}'")
        except Exception as e:
            logging.error(f"画像メタ情報抽出失敗: {file_path}, err={e}")
            return

        # stable diffusion 配下のサブフォルダ作成 or 既存使用
        target_folder_id = find_or_create_subfolder(
            parent_id=self.stable_folder_id, subfolder_name=date_dir
        )

        # 画像を Eagle に登録
        item = api_item.EAGLE_ITEM_PATH(
            filefullpath=file_path,
            filename=os.path.basename(file_path),
            annotation=annotation,
            tags=tags,
        )
        resp = api_item.add_from_path(
            item=item,
            folderId=target_folder_id,
            server_url=EAGLE_SERVER_URL,
            port=EAGLE_SERVER_PORT,
        )
        if resp.status_code == 200:
            logging.info(f"Eagle 転送成功: {file_path}")
        else:
            logging.error(
                f"Eagle 転送失敗: {file_path}, status={resp.status_code}, text={resp.text}"
            )

        save_processed_hash(file_hash)

    # イベント発生時はスレッドプールにて非同期処理
    def on_created(self, event):
        if not event.is_directory:
            self.executor.submit(self.process_file, event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            self.executor.submit(self.process_file, event.src_path)

    def on_moved(self, event):
        if not event.is_directory:
            self.executor.submit(self.process_file, event.dest_path)


# 初回スキャンも並列実行する例
def initial_scan(folder_list, handler):
    futures = []
    for fol in folder_list:
        for root, dirs, files in os.walk(fol):
            for fn in files:
                if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                    fpath = os.path.join(root, fn)
                    futures.append(handler.executor.submit(handler.process_file, fpath))
    if futures:
        concurrent.futures.wait(futures)


# ------------------------------------------------------------------------
# 6) メインエントリ
# ------------------------------------------------------------------------
def main():
    # A) 監視フォルダ (例: 環境変数 EAGLE_GOOGLE_DRIVE_FOLDER1=... など)
    folder_list = []
    for k, v in os.environ.items():
        if k.startswith("EAGLE_GOOGLE_DRIVE_FOLDER") and v:
            for path_ in v.split(","):
                path_ = path_.strip()
                if path_:
                    folder_list.append(path_)

    if not folder_list:
        logging.error("EAGLE_GOOGLE_DRIVE_FOLDER* の環境変数が設定されていません。")
        sys.exit(1)

    valid_folders = []
    for f_ in folder_list:
        if os.path.exists(f_):
            valid_folders.append(os.path.normpath(f_))
        else:
            logging.error(f"監視対象フォルダが存在しません: {f_}")
    if not valid_folders:
        logging.error("監視対象フォルダが一つも有効ではありません。終了します。")
        sys.exit(1)

    # B) stable diffusion フォルダID の取得
    stable_diff_folder_id = fetch_or_create_stable_diffusion_folder()
    if not stable_diff_folder_id:
        logging.error(f"'{STABLE_DIFFUSION_NAME}' フォルダID を取得できず。終了。")
        sys.exit(1)

    # C) Watchdog 開始
    handler = NewFileHandler(valid_folders, stable_diff_folder_id)
    observer = Observer()
    for vf in valid_folders:
        observer.schedule(handler, vf, recursive=True)
        logging.info(f"監視開始: {vf}")
    observer.start()

    # D) 初回スキャン (既存ファイルも並列処理で実施)
    initial_scan(valid_folders, handler)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt: 監視停止中...")
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
