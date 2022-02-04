"""
ARGO module made for some specific usage in downloading
data from the GDAC center using rsync and screen
"""
import os
from typing import Optional

import pandas as pd


def build_dl(argo_df: pd.DataFrame, ARGO_localFTP: Optional[str] = None):
    """
    Build the download command using rsync and screen
    """
    print("\nBuilding download list")
    dac_floats = pd.DataFrame(
        argo_df.file.str.split("/").str[:2].str.join("/").unique(),
        columns=["combined"],
    )
    dac_floats["dac"] = dac_floats.combined.str.split("/").str[0]
    dac_floats["float"] = dac_floats.combined.str.split("/").str[1]

    if ARGO_localFTP is None:
        ARGO_localFTP = "/data/datos/ARGO/gdac"

    cmd_template = "screen -dmS auto_{}_{}_{:%Y%m%d_%Hh} rsync -avvzhP --delete-during --timeout=30 vdmzrs.ifremer.fr::argo/{} {}"

    today = pd.Timestamp.now(tz="America/Lima")
    dl_list = [
        cmd_template.format(
            row.dac,
            row.float,
            today,
            row.combined,
            os.path.join(ARGO_localFTP, "dac", row.dac),
        )
        for _, row in dac_floats.iterrows()
    ]
    print("Done\n")
    return dl_list


def launch_shell(cmd_list):
    """
    Launch the built command from a subshell
    """
    print("\nWriting and launching shell script")
    with open("launch_shell.sh", "w") as shfile:
        shfile.write("#!/bin/bash -l\n\n")
        shfile.write("\n".join(cmd_list))
        shfile.write("\n\nwhile screen -list | grep -q auto\ndo\n    sleep 1\ndone")
    os.system("sh launch_shell.sh")
    print("Done\n")
