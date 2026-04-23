import requests
import boto3
import subprocess
import pdb

METADATA_BASE = "http://169.254.169.254/latest"


def get_imdsv2_token():
    r = requests.put(
        f"{METADATA_BASE}/api/token",
        headers={
            "X-aws-ec2-metadata-token-ttl-seconds": "21600"
        },
        timeout=2,
    )
    r.raise_for_status()
    return r.text


def metadata(path, token):
    r = requests.get(
        f"{METADATA_BASE}/meta-data/{path}",
        headers={
            "X-aws-ec2-metadata-token": token
        },
        timeout=2,
    )
    r.raise_for_status()
    return r.text

pdb.set_trace()
token = get_imdsv2_token()
instance_id = metadata("instance-id", token)
region = metadata("placement/region", token)
ec2 = boto3.client("ec2", region_name=region)

try:
    print("Starting job")
except Exception:
    raise
else:
    print(f"Stopping {instance_id}")
    ec2.stop_instances(InstanceIds=[instance_id])