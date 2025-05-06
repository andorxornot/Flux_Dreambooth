from huggingface_hub import snapshot_download

local_dir = "./3d_icon"
snapshot_download(
    "LinoyTsaban/3d_icon",
    local_dir=local_dir, repo_type="dataset",
    ignore_patterns=".gitattributes",
)