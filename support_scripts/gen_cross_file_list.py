#!/usr/bin/env python

import renderapi
import argparse
import os
import gzip
import json


def gen_file_list(cross, stack, base_path, n, match, ppf, logdir, render_connect_params, **kwargs):

    optflow = defaults(n, **kwargs)

    optflow["host"] = render_connect_params["host"]
    optflow["port"] = render_connect_params["port"]
    optflow["matchCollection"] = match
    optflow["owner"] = render_connect_params["owner"]
    render = renderapi.connect(**render_connect_params)
    tilespecs = renderapi.tilespec.get_tile_specs_from_stack(
        stack, render=render)
    imageurls = {i.tileId: i.ip[0].imageUrl.split(":")[-1] for i in tilespecs}

    with gzip.open(cross) as f:
        pairs = json.loads(f.read().decode("ascii"))

    sub_pair_list = [pairs["neighborPairs"][i:i+ppf]
                     for i in range(0, len(pairs["neighborPairs"]), ppf)]

    N_dict = {}
    for count, sub_pairs in enumerate(sub_pair_list):
        optflow["images"] = []
        for pair in sub_pairs:
            if logdir is not None:
                if imageurls[pair["p"]["id"]] not in N_dict:
                    with open(logpath(logdir, imageurls[pair["p"]["id"]]), "r") as f:
                        N_dict[imageurls[pair["p"]["id"]]] = float(
                            next(f).split(" ")[0])
                if imageurls[pair["q"]["id"]] not in N_dict:
                    with open(logpath(logdir, imageurls[pair["q"]["id"]]), "r") as f:
                        N_dict[imageurls[pair["q"]["id"]]] = float(
                            next(f).split(" ")[0])
            if sub_list in kwargs:
                if (int(pair["p"]["pGroupId"]) not in kwargs[sub_list]) and (int(pair["q"]["qGroupId"]) not in kwargs[sub_list]):
                    continue
            im_data = {}
            im_data["p"] = imageurls[pair["p"]["id"]]
            im_data["q"] = imageurls[pair["q"]["id"]]
            im_data["pId"] = pair["p"]["id"]
            im_data["qId"] = pair["q"]["id"]
            im_data["pGroupId"] = pair["p"]["groupId"]
            im_data["qGroupId"] = pair["q"]["groupId"]
            col_p = int(imageurls[pair["p"]["id"]].split("-")[-2])
            col_q = int(imageurls[pair["p"]["id"]].split("-")[-2])
            if logdir is not None:
                if "pGroupId" != "1.0" and "qGroupId" != "1.0":
                    if (N_dict[imageurls[pair["p"]["id"]]] - col_p == 1) or (N_dict[imageurls[pair["q"]["id"]]] - col_q == 1):
                        im_data["features"] = kwargs.get("features", 2)
            optflow["images"].append(im_data)

        with gzip.GzipFile(base_path+"_%d.json.gz" % (count), "w") as fout:
            fout.write(json.dumps(optflow).encode('utf-8'))


def logpath(log_dir, imageurl):
    image_name = imageurl.split("/")[-1]
    image_name = "-".join(image_name.split("-")[:-1])  # Strip -InLens
    log_name = image_name+".log"
    lpath = log_dir + "/" + log_name
    return lpath


def defaults(n, **kwargs):
    d = {}
    d["style"] = kwargs.get("style", 1)
    d["debug"] = kwargs.get("debug", False)
    if "features" in kwargs:
        d["features"] = kwargs.get("features", 2)
    d["homo"] = kwargs.get("homo", 4)
    d["ratio"] = kwargs.get("ratio", 0.7)
    d["ransac"] = kwargs.get("ransac", 5)
    d["hessianThreshold"] = kwargs.get("hessianThreshold", 1600)
    d["scale"] = kwargs.get("scale", 0.5)
    d["output_dir"] = kwargs.get("output_dir", ".")
    if "top" in kwargs:
        if "rois" not in d:
            d["rois"] = {}
        if kwargs["top"]:
            d["rois"]["top"] = kwargs["top"]
    if "bottom" in kwargs:
        if "rois" not in d:
            d["rois"] = {}
        if kwargs["bottom"]:
            d["rois"]["bottom"] = kwargs["bottom"]
    d["output_type"] = kwargs.get("output_type", "random_points")
    d["npoints"] = kwargs.get("npoints", n)
    return d


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cross", type=str)
    parser.add_argument("--stack", default="v1_acquire",
                        type=str)
    parser.add_argument("--base_path", default="/tmp/optflow", type=str)
    parser.add_argument("--n", default=10, type=int, help="Number of points")
    parser.add_argument("--match", default="forgetful_owner", type=str)
    parser.add_argument("--project", default=os.environ.get("RENDER_PROJECT"),
                        type=str)
    parser.add_argument("--owner", default=os.environ.get("RENDER_OWNER"),
                        type=str)
    parser.add_argument("--host", default=os.environ.get("RENDER_HOST"),
                        type=str)
    parser.add_argument("--port", default=os.environ.get("RENDER_PORT"),
                        type=str)
    parser.add_argument("--ppf", default=5000, type=int,
                        help="Pairs per file, note last file may be smaller.")
    parser.add_argument("--client_scripts",
                        default=os.environ.get("RENDER_CLIENT_SCRIPTS"),
                        type=str)
    parser.add_argument("--top", default=0, type=int)
    parser.add_argument("--bottom", default=0, type=int)
    parser.add_argument("--memGB", default=os.environ.get("RENDER_CLIENT_HEAP"),
                        type=str)
    parser.add_argument("--logdir", type=str)

    args = parser.parse_args()

    render_connect_params = {
        "host": args.host,
        "port": args.port,
        "owner": args.owner,
        "project": args.project,
        "client_scripts": args.client_scripts,
        "memGB": args.memGB
    }

    gen_file_list(args.cross, args.stack, args.base_path,
                  args.n, args.match, args.ppf, args.logdir, render_connect_params, top=args.top, bottom=args.bottom)
