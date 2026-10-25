import anthropic
from dotenv import load_dotenv

load_dotenv()

try:
    client = anthropic.Anthropic()
    r = client.messages.create(
        model='claude-haiku-4-5',
        max_tokens=32,
        messages=[{'role': 'user', 'content': 'Reply with just: API key working'}]
    )
    print(r.content[0].text)
except anthropic.APIStatusError as e:
    print(f"Failed: {e.body.get('error', {}).get('message', str(e))}")
except Exception as e:
    print(f"Failed: {getattr(e, 'message', str(e)).strip(chr(34))}")
