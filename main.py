import os

from flask import Flask, request, abort
from dotenv import load_dotenv

from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage,
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent

import utils
import pandas as pd
from keras.models import load_model

app = Flask(__name__)

load_dotenv()

configuration = Configuration(access_token=os.getenv("CHANNEL_ACCESS_TOKEN"))
handler = WebhookHandler(os.getenv("CHANNEL_SECRET"))

model = load_model("./model/lstm_model.h5")


@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers["X-Line-Signature"]

    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        app.logger.info(
            "Invalid signature. Please check your channel access token/channel secret."
        )
        abort(400)

    return "OK"


@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    with ApiClient(configuration) as api_client:
        dataframeToPredict = pd.DataFrame({"message": ["" + event.message.text]})
        predicted_labels = utils.predict_class(
            model=model, dataframeToPredict=dataframeToPredict
        )

        reply = f"ข้อความ: {event.message.text}\nผลการทำนาย: {predicted_labels[0]}"

        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=reply)],
            )
        )


if __name__ == "__main__":
    app.run("0.0.0.0", port=8080, debug=False)
