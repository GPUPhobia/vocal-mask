import requests
from discord import Webhook, RequestsWebhookAdapter
import json

with open("discord_config.json", "r") as inf:
    discord_config = json.load(inf)

def send_message(msg):
    webhook_id = discord_config["webhook_id"]
    token = discord_config["token"]
    webhook = Webhook.partial(webhook_id, token, adapter=RequestsWebhookAdapter())
    webhook.send(msg, username="paperspace")
