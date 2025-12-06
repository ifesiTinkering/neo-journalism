from dome_api_sdk import DomeClient

dome = DomeClient({"api_key": "your-api-key-here"})

market_price = dome.polymarket.markets.get_market_price({
    "token_id": "98250445447699368679516529207365255018790721464590833209064266254238063117329"
})
print(f"Market Price: {market_price.price}")