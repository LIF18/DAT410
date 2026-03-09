import difflib

# External Database 
DATABASE = {
    "weather": {
        "gothenburg": "Raining, 4°C, strong winds",
        "stockholm": "Cloudy, 2°C",
        "kiruna": "Snowing, -10°C"
    },
    "restaurant": {
        "pizza": "Sannegårdens Pizzeria",
        "spicy": "Spicy Duck Head & Neck Diner on Avenyn",
        "sushi": "Super Sushi near Korsvägen",
        "vegan": "Blackbird"
    },
    "transit": {
        "chalmers": "Tram 6 to Chalmers Campus arriving in 3 mins",
        "central": "Bus 16 to Centralstation arriving in 5 mins",
        "airport": "Flygbussarna departing in 10 mins"
    }
}


# Fuzzy Matching Helper
def fuzzy_extract(user_input, valid_keys, cutoff=0.7):
    """
    Splits user text and finds the closest match to valid_keys.
    cutoff=0.7 means it tolerates minor typos (e.g., "chalmerss" -> "chalmers").
    """
    words = user_input.lower().split()
    for word in words:
        # get_close_matches returns a list of matches. [0] is the best match.
        matches = difflib.get_close_matches(word, valid_keys, n=1, cutoff=cutoff)
        if matches:
            return matches[0] # Return the correctly spelled key from our DB
    return None


# Task Framework
class Task:
    def __init__(self):
        self.slots = {}
        
    def is_complete(self):
        return all(value is not None for value in self.slots.values())
    
    def get_prompt(self): pass
    def extract_info(self, user_input): pass
    def execute(self): pass

class WeatherTask(Task):
    def __init__(self):
        super().__init__()
        self.slots = {"city": None}

    def extract_info(self, user_input):
        # Dynamically get valid cities from the database keys
        valid_cities = list(DATABASE["weather"].keys())
        match = fuzzy_extract(user_input, valid_cities)
        if match:
            self.slots["city"] = match

    def get_prompt(self):
        if self.slots["city"] is None:
            return "Which city do you want the weather for? (e.g., Gothenburg, Stockholm)"

    def execute(self):
        city = self.slots["city"]
        weather = DATABASE["weather"][city]
        return f"The weather in {city.capitalize()} is currently: {weather}."

class RestaurantTask(Task):
    def __init__(self):
        super().__init__()
        self.slots = {"food_type": None}

    def extract_info(self, user_input):
        valid_foods = list(DATABASE["restaurant"].keys())
        match = fuzzy_extract(user_input, valid_foods)
        if match:
            self.slots["food_type"] = match

    def get_prompt(self):
        if self.slots["food_type"] is None:
            return "What kind of food are you looking for? (e.g., spicy, pizza, sushi)"

    def execute(self):
        food = self.slots["food_type"]
        restaurant = DATABASE["restaurant"][food]
        return f"I recommend '{restaurant}' for some great {food} food!"

class TransitTask(Task):
    def __init__(self):
        super().__init__()
        self.slots = {"destination": None}

    def extract_info(self, user_input):
        valid_dests = list(DATABASE["transit"].keys())
        match = fuzzy_extract(user_input, valid_dests)
        if match:
            self.slots["destination"] = match

    def get_prompt(self):
        if self.slots["destination"] is None:
            return "Where are you heading? (e.g., Chalmers, Central, Airport)"

    def execute(self):
        dest = self.slots["destination"]
        transit = DATABASE["transit"][dest]
        return f"To get to {dest.capitalize()}: {transit}."
    

# Dialogue Manager
class DialogueManager:
    def __init__(self):
        self.current_task = None

    def route_intent(self, user_input):
        # We can also use fuzzy matching for intents!
        intent_map = {
            "weather": WeatherTask, "forecast": WeatherTask,
            "restaurant": RestaurantTask, "food": RestaurantTask, "hungry": RestaurantTask,
            "tram": TransitTask, "bus": TransitTask, "transit": TransitTask
        }
        match = fuzzy_extract(user_input, list(intent_map.keys()), cutoff=0.8)
        if match:
            return intent_map[match]()
        return None

    def process_input(self, user_input):
        if user_input.lower() in ["exit", "quit"]: return "Goodbye!"

        if self.current_task is None:
            self.current_task = self.route_intent(user_input)
            if self.current_task is None:
                return "I can help with weather, restaurants, or transit. What do you need?"

        self.current_task.extract_info(user_input)

        if self.current_task.is_complete():
            response = self.current_task.execute()
            self.current_task = None 
            return response
        else:
            return self.current_task.get_prompt()

if __name__ == "__main__":
    bot = DialogueManager()
    print("Assistant: Hello! I can help with weather, restaurants, and transit. (Type 'exit' to quit)")
    while True:
        user_text = input("You: ")
        reply = bot.process_input(user_text)
        print(f"Assistant: {reply}")
        if reply == "Goodbye!": break