# Usage
Add the python libraries you need into `requirements.txt` and run `make upload_env`.

This will upload a packed virtual python environment to your home folder on hdfs.
The code in `main.py` can then be run on a gpu node on the cluster by running the command:
`skein application submit gpu.yaml` .

Output can be found in yarn:***   ***
or by running ``yarn logs -applicationId $APP_ID -log_files application.driver.log`` where `$APP_ID` is the output from the 
`skein application submit gpu.yaml` command.
