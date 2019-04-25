import requests
from discord import Webhook, RequestsWebhookAdapter

def send_message(msg):
    webhook_id = 571049429549842453
    token = "3hWa6QVIMHBbryK2psgEcMl8QXx36X41Ryt279HPDtCI0mxMnoVSP8XmUwcYH-0shbaQ" 
    webhook = Webhook.partial(webhook_id, token, adapter=RequestsWebhookAdapter())
    webhook.send(msg, username="paperspace")
