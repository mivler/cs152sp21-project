from bing_image_downloader import downloader


categories = ["Impressionist art", "Cubist art", "Contemporary art", "Modern Art", "Abstract art", "Surrealist art", "Xieyi art", "Gongbi art", "min-hwa art", "Mughal art", "Kente art"]


for query_string in categories:
    out_dir = "dataset"
    downloader.download(query_string, limit=300, output_dir=out_dir, force_replace=True)
