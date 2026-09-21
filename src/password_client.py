import requests


# =========================
# SERVER SETTINGS
# =========================

SERVER_URL = "http://192.168.1.6/"


def fetch_password_fingers():
    """
    Fetch the required password from the HTTP server.

    Server currently returns plain text like:
        [0,0,0,0,1]

    Returns:
        list of 5 integers:
        [thumb, index, middle, ring, pinky]
    """

    try:
        response = requests.get(SERVER_URL, timeout=5)
        response.raise_for_status()

        text = response.text.strip()

        # Convert "[0,0,0,0,1]" into [0, 0, 0, 0, 1]
        text = text.replace("[", "").replace("]", "").replace(" ", "")
        fingers = text.split(",")

        fingers = [int(x) for x in fingers]

        if len(fingers) != 5:
            print("Invalid password length:", fingers)
            return None

        if not all(value in [0, 1] for value in fingers):
            print("Invalid password values:", fingers)
            return None

        print("Fetched password fingers:", fingers)
        return fingers

    except requests.exceptions.Timeout:
        print("Password server timeout.")
        return None

    except requests.exceptions.ConnectionError:
        print("Could not connect to password server.")
        return None

    except Exception as e:
        print("Could not fetch password from server:", e)
        return None


def fingers_to_text(fingers):
    names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

    raised = []

    for name, value in zip(names, fingers):
        if value == 1:
            raised.append(name)

    if len(raised) == 0:
        return "No fingers raised"

    return "Raised fingers: " + ", ".join(raised)


if __name__ == "__main__":
    password = fetch_password_fingers()

    if password is not None:
        print("Finger array:", password)
        print("Text:", fingers_to_text(password))