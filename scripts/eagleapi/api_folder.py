# see also https://api.eagle.cool/folder/list
#
import requests
import sys

from . import api_util


def create(
    newfoldername,
    server_url="http://localhost",
    port=41595,
    allow_duplicate_name=True,
    timeout_connect=3,
    timeout_read=10,
):
    """EAGLE API:/api/folder/list

    Method: POST

    Returns:
        list(response dict): return list of response.json()
    """
    API_URL = f"{server_url}:{port}/api/folder/create"

    def _init_data(newfoldername):
        _data = {}
        if newfoldername and newfoldername != "":
            _data.update({"folderName": newfoldername})
        return _data

    data = _init_data(newfoldername)

    # check duplicate if needed
    if not allow_duplicate_name:
        r_post = list()
        _ret = api_util.findFolderByName(r_post, newfoldername)
        if _ret != None or len(_ret) > 0:
            print(
                f'ERROR: create folder with same name is forbidden by option. [eagleapi.folder.create] foldername="{newfoldername}"',
                file=sys.stderr,
            )
            return

    r_post = requests.post(API_URL, json=data, timeout=(timeout_connect, timeout_read))
    return r_post


def create_subfolder(
    newfoldername,
    parent_id,
    server_url="http://localhost",
    port=41595,
    allow_duplicate_name=True,
    timeout_connect=3,
    timeout_read=10,
):
    """
    Eagle API: /api/folder/create
    parent_id が指定されていればサブフォルダとして作成する。

    このサンプルでは、親フォルダは "stable diffusion" フォルダを想定。
    重複チェックは extendTags に 'stable diffusion' が含まれる & フォルダ名一致 として判定。
    """
    API_URL = f"{server_url}:{port}/api/folder/create"

    data = {"folderName": newfoldername}
    if parent_id:
        data["parent"] = parent_id

    # -------------------------------------------------------
    # allow_duplicate_name=False のときは重複をチェック (extendTags+name)
    # -------------------------------------------------------
    if not allow_duplicate_name:
        resp_list = list(server_url=server_url, port=port)
        if resp_list.status_code != 200:
            print("ERROR: cannot get folder list", file=sys.stderr)
            return resp_list

        existing = api_util.findFolderByNameAndExtendTag(
            resp_list, "stable diffusion", newfoldername
        )
        if existing is not None:
            print(
                f'ERROR: extendTags に "stable diffusion" があり、かつ同名フォルダ "{newfoldername}" が既に存在します。',
                file=sys.stderr,
            )
            # 400系など適当なエラーを返したい場合:
            r_fake = requests.models.Response()
            r_fake.status_code = 400
            r_fake._content = b'{"error":"Folder already exists"}'
            return r_fake

    # 新規作成
    r_post = requests.post(API_URL, json=data, timeout=(timeout_connect, timeout_read))
    return r_post


def rename(
    folderId,
    newName,
    server_url="http://localhost",
    port=41595,
    timeout_connect=3,
    timeout_read=10,
):
    """EAGLE API:/api/folder/rename

    Method: POST

    Returns:
        list(response dict): return list of response.json()
    """
    data = {"folderId": folderId, "newName": newName}
    API_URL = f"{server_url}:{port}/api/folder/rename"
    r_post = requests.post(API_URL, json=data, timeout=(timeout_connect, timeout_read))
    return r_post


def list(server_url="http://localhost", port=41595, timeout_connect=3, timeout_read=10):
    """EAGLE API:/api/folder/list

    Method: GET

    Returns:
        Response: return of requests.post
    """

    API_URL = f"{server_url}:{port}/api/folder/list"

    r_get = requests.get(API_URL, timeout=(timeout_connect, timeout_read))

    return r_get
