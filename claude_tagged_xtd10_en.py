import os 
import pandas as pd
objectSamples = [
        {"id": 1, "text": "a cat with its paws on a computer mouse at a desk", "feature": "Objects/Entities", "signals": "cat, computer mouse, desk", "confidence": "High", "notes": "Multiple concrete objects clearly identified"},
        {"id": 11, "text": "a very nice looking motorcycle parked in a lot by a car", "feature": "Objects/Entities", "signals": "motorcycle, car", "confidence": "High", "notes": "Vehicle entities with descriptive attributes"},
        {"id": 18, "text": "a bird perched on top of a banana tree with lots of leaves", "feature": "Objects/Entities", "signals": "bird, banana tree, leaves", "confidence": "High", "notes": "Natural objects and living entities"},
        {"id": 33, "text": "a pizza on a table with a bowl of grapes and drinks", "feature": "Objects/Entities", "signals": "pizza, table, bowl, grapes, drinks", "confidence": "High", "notes": "Food objects and containers"},
        {"id": 41, "text": "a laptop computer sitting on a computer desk next to a keyboard", "feature": "Objects/Entities", "signals": "laptop computer, computer desk, keyboard", "confidence": "High", "notes": "Technology objects and furniture"},
        {"id": 48, "text": "the big bowl of fruit contains pomegranate, kiwis, lemons, limes, and a cantaloupe", "feature": "Objects/Entities", "signals": "bowl, pomegranate, kiwis, lemons, limes, cantaloupe", "confidence": "High", "notes": "Container and multiple fruit entities"},
        {"id": 71, "text": "a brown and white horse standing on top of a grass field", "feature": "Objects/Entities", "signals": "horse, grass field", "confidence": "High", "notes": "Animal entity with environmental object"},
        {"id": 99, "text": "a cute little brown teddy bear sits on a rock by a bush", "feature": "Objects/Entities", "signals": "teddy bear, rock, bush", "confidence": "High", "notes": "Toy object with natural elements"},
        {"id": 127, "text": "a man's watch displayed on a leaf on a wooden surface", "feature": "Objects/Entities", "signals": "watch, leaf, wooden surface", "confidence": "High", "notes": "Accessory object with natural and artificial surfaces"},
        {"id": 158, "text": "a teddy bear sitting on the ground, looking at the camera", "feature": "Objects/Entities", "signals": "teddy bear, ground, camera", "confidence": "High", "notes": "Toy entity with implied technology object"},
        {"id": 186, "text": "a large clay planter with a tree and purple flowers inside of it", "feature": "Objects/Entities", "signals": "clay planter, tree, purple flowers", "confidence": "High", "notes": "Container with plant entities"},
        {"id": 258, "text": "glass vases with flowers and water are on a granite table", "feature": "Objects/Entities", "signals": "glass vases, flowers, water, granite table", "confidence": "High", "notes": "Multiple containers and natural elements"},
        {"id": 293, "text": "a head of cabbage and carrots on a cutting board", "feature": "Objects/Entities", "signals": "cabbage, carrots, cutting board", "confidence": "High", "notes": "Vegetable entities with kitchen tool"},
        {"id": 321, "text": "a toothbrush is sitting on a sink that has the words mystery toothbrush on it", "feature": "Objects/Entities", "signals": "toothbrush, sink", "confidence": "High", "notes": "Personal care object with bathroom fixture"},
        {"id": 367, "text": "the shop features a large collection of many types, sizes, and colors of teddy bears", "feature": "Objects/Entities", "signals": "shop, teddy bears", "confidence": "High", "notes": "Commercial space with toy collection"},
        {"id": 424, "text": "an orange tree laden with ripe fruits in front of a house", "feature": "Objects/Entities", "signals": "orange tree, fruits, house", "confidence": "High", "notes": "Natural entity with architectural object"},
        {"id": 471, "text": "a pink cellphone in front of a mirror and a toy cow", "feature": "Objects/Entities", "signals": "cellphone, mirror, toy cow", "confidence": "High", "notes": "Technology, household, and toy objects"},
        {"id": 532, "text": "an old red coke cola machine on the side of a building", "feature": "Objects/Entities", "signals": "coke cola machine, building", "confidence": "High", "notes": "Vintage commercial object with architecture"},
        {"id": 620, "text": "a pair of red and white scissors sitting on top of a metal counter", "feature": "Objects/Entities", "signals": "scissors, metal counter", "confidence": "High", "notes": "Tool object with surface material"},
        {"id": 730, "text": "a bowl of powdered donuts sitting next to a bowl of chocolate donuts", "feature": "Objects/Entities", "signals": "bowl, powdered donuts, chocolate donuts", "confidence": "High", "notes": "Containers with different food varieties"},
        {"id": 790, "text": "a singapore airlines commercial aircraft landing on the runway next to the water", "feature": "Objects/Entities", "signals": "aircraft, runway, water", "confidence": "High", "notes": "Large vehicle with infrastructure and natural element"},
        {"id": 836, "text": "a grey cat sniffing an item that is put in front of him by a person", "feature": "Objects/Entities", "signals": "cat, item", "confidence": "Medium", "notes": "Animal entity with unspecified object"},
        {"id": 890, "text": "a close up picture of some tangerines in a bowl", "feature": "Objects/Entities", "signals": "tangerines, bowl", "confidence": "High", "notes": "Fruit entities in container"},
        {"id": 925, "text": "a couple of parking meters sitting on top of a patch of grass", "feature": "Objects/Entities", "signals": "parking meters, grass", "confidence": "High", "notes": "Urban infrastructure with natural surface"},
        {"id": 976, "text": "a small stack of antique leather bound books with a pair of glasses and a pocket watch", "feature": "Objects/Entities", "signals": "books, glasses, pocket watch", "confidence": "High", "notes": "Vintage objects with accessories"}
    ];

actionSamples = [
        {"id": 0, "text": "major league baseball game with player from pittsburgh pirates crossing home plate", "feature": "Actions/Events", "signals": "crossing, baseball game", "confidence": "High", "notes": "Sports action with specific movement verb"},
        {"id": 2, "text": "flock of birds taking off from water near long pier on open ocean", "feature": "Actions/Events", "signals": "taking off", "confidence": "High", "notes": "Flight action with directional movement"},
        {"id": 10, "text": "a man riding a wave on top of a surfboard in the ocean", "feature": "Actions/Events", "signals": "riding", "confidence": "High", "notes": "Surfing action with continuous aspect"},
        {"id": 25, "text": "a man on a tennis court about to swipe at a tennis ball", "feature": "Actions/Events", "signals": "about to swipe", "confidence": "High", "notes": "Imminent action with sports context"},
        {"id": 26, "text": "a skier soars through the air as a crowd looks on", "feature": "Actions/Events", "signals": "soars, looks on", "confidence": "High", "notes": "Multiple actions - movement and observation"},
        {"id": 44, "text": "a man grabbing a bunch of bananas from a large display of containers", "feature": "Actions/Events", "signals": "grabbing", "confidence": "High", "notes": "Acquisition action with directional movement"},
        {"id": 64, "text": "a snowboarder hits the ground and is in the process of falling forward", "feature": "Actions/Events", "signals": "hits, falling", "confidence": "High", "notes": "Impact and progressive action"},
        {"id": 84, "text": "the man is riding his skate board around the cones", "feature": "Actions/Events", "signals": "riding around", "confidence": "High", "notes": "Continuous motion with spatial path"},
        {"id": 95, "text": "the tennis player is running to hit the ball with his racket", "feature": "Actions/Events", "signals": "running to hit", "confidence": "High", "notes": "Purposive movement toward action"},
        {"id": 115, "text": "a boy is running up a hill holding a kite in the air", "feature": "Actions/Events", "signals": "running up, holding", "confidence": "High", "notes": "Directional movement with simultaneous action"},
        {"id": 136, "text": "police bend down to put their arms on a man who is falling to the ground off of his motorcycle in the middle of a crowded street", "feature": "Actions/Events", "signals": "bend down, put, falling", "confidence": "High", "notes": "Complex multi-agent action sequence"},
        {"id": 148, "text": "a guy catches a frisbee as another looks on during a frisbee tournament beneath cloudy skies", "feature": "Actions/Events", "signals": "catches, looks on", "confidence": "High", "notes": "Catching action with observation in sports context"},
        {"id": 168, "text": "a man pulling two cows by ropes with a lot of people gathered together", "feature": "Actions/Events", "signals": "pulling, gathered", "confidence": "High", "notes": "Force application and social assembly"},
        {"id": 172, "text": "a man flipping in the air with a snowboard above a snow covered hill", "feature": "Actions/Events", "signals": "flipping", "confidence": "High", "notes": "Aerial acrobatic action"},
        {"id": 178, "text": "a man holding a bat being watched by a crowd of people", "feature": "Actions/Events", "signals": "holding, being watched", "confidence": "High", "notes": "Possession action with passive observation"},
        {"id": 213, "text": "a woman standing under an umbrella extends her hand to catch raindrops", "feature": "Actions/Events", "signals": "standing, extends, catch", "confidence": "High", "notes": "Multiple coordinated actions"},
        {"id": 243, "text": "a man riding a skateboard up the side of a ramp", "feature": "Actions/Events", "signals": "riding up", "confidence": "High", "notes": "Directional movement on equipment"},
        {"id": 280, "text": "a little boy using a shovel to clear a path in the snow", "feature": "Actions/Events", "signals": "using, clear", "confidence": "High", "notes": "Tool use for purposive action"},
        {"id": 295, "text": "a group of men try to push a large truck onto a road", "feature": "Actions/Events", "signals": "try to push", "confidence": "High", "notes": "Attempted force application by group"},
        {"id": 322, "text": "the baseball player is in the middle of batting at a pitch", "feature": "Actions/Events", "signals": "batting", "confidence": "High", "notes": "Progressive sports action"},
        {"id": 416, "text": "a man riding a skateboard through he air over a walkway", "feature": "Actions/Events", "signals": "riding through", "confidence": "High", "notes": "Aerial movement with path"},
        {"id": 434, "text": "a man catches a wave on his surfboard and holds his arms up to balance", "feature": "Actions/Events", "signals": "catches, holds up, balance", "confidence": "High", "notes": "Multiple coordinated surfing actions"},
        {"id": 542, "text": "a boy is swinging at a ball during a baseball game", "feature": "Actions/Events", "signals": "swinging at", "confidence": "High", "notes": "Contact attempt action in sports"},
        {"id": 648, "text": "a man throws a tennis ball in the air to serve it", "feature": "Actions/Events", "signals": "throws, serve", "confidence": "High", "notes": "Preparatory and main tennis actions"},
        {"id": 913, "text": "man in mid air reaching between his legs to reach a frisbee", "feature": "Actions/Events", "signals": "reaching", "confidence": "High", "notes": "Complex aerial reaching action"}
    ];


sceneSamples = [
        {"id": 16, "text": "at nighttime, a car sits at a red light in an asian city", "feature": "Scene Type/Setting", "signals": "nighttime, asian city", "confidence": "High", "notes": "Urban nighttime scene with cultural specificity"},
        {"id": 31, "text": "a view of a city street with several double decker buses parked along the side as cars drive by", "feature": "Scene Type/Setting", "signals": "city street", "confidence": "High", "notes": "Urban street scene with traffic"},
        {"id": 42, "text": "photo taken from vehicle on mountain narrow mountain pass looking very dangerous with high steep drop down", "feature": "Scene Type/Setting", "signals": "mountain narrow mountain pass", "confidence": "High", "notes": "Dangerous mountain road scene"},
        {"id": 63, "text": "a group of pedestrians walking past a subway train at a subway station", "feature": "Scene Type/Setting", "signals": "subway station", "confidence": "High", "notes": "Underground transit scene"},
        {"id": 87, "text": "an old mediterranean yellow mosaic tiled bathroom with ornate hanging light", "feature": "Scene Type/Setting", "signals": "mediterranean bathroom", "confidence": "High", "notes": "Cultural architectural interior scene"},
        {"id": 112, "text": "a walkway between rows of pews in a large church", "feature": "Scene Type/Setting", "signals": "large church", "confidence": "High", "notes": "Religious interior architecture"},
        {"id": 126, "text": "a dark room with a large window and no curtain, an easel and assorted furniture", "feature": "Scene Type/Setting", "signals": "dark room, art studio", "confidence": "High", "notes": "Artist workspace scene"},
        {"id": 137, "text": "large angle shot of a house that shows a living room, dinning room, and kitchen", "feature": "Scene Type/Setting", "signals": "house interior, multiple rooms", "confidence": "High", "notes": "Domestic multi-room interior"},
        {"id": 139, "text": "travelers gathered around a carousel in baggage claim at an airport", "feature": "Scene Type/Setting", "signals": "baggage claim, airport", "confidence": "High", "notes": "Travel terminal scene"},
        {"id": 142, "text": "a man standing near four dump trucks in a quarry at night", "feature": "Scene Type/Setting", "signals": "quarry at night", "confidence": "High", "notes": "Industrial nighttime scene"},
        {"id": 144, "text": "outside picture of a city with the beach and kites flying in the sky", "feature": "Scene Type/Setting", "signals": "city with beach", "confidence": "High", "notes": "Urban coastal scene"},
        {"id": 174, "text": "an interior of living area including sofa, table, and shelves", "feature": "Scene Type/Setting", "signals": "living area interior", "confidence": "High", "notes": "Domestic interior scene"},
        {"id": 193, "text": "a coffee shop with chairs and tables outside of the shop", "feature": "Scene Type/Setting", "signals": "coffee shop exterior", "confidence": "High", "notes": "Commercial outdoor dining scene"},
        {"id": 273, "text": "view of part of wall of kitchen, with refrigerator, stove, overhead microwave, cabinets and counters", "feature": "Scene Type/Setting", "signals": "kitchen interior", "confidence": "High", "notes": "Domestic cooking space"},
        {"id": 327, "text": "a kitchen with a pantry and a oven inside of it", "feature": "Scene Type/Setting", "signals": "kitchen interior", "confidence": "High", "notes": "Food preparation space"},
        {"id": 349, "text": "household bathroom with colorful art on floor tiles and similar shower curtain", "feature": "Scene Type/Setting", "signals": "household bathroom", "confidence": "High", "notes": "Decorated domestic bathroom"},
        {"id": 382, "text": "a den with large windows, chairs, a couch, table and a television", "feature": "Scene Type/Setting", "signals": "den interior", "confidence": "High", "notes": "Relaxation room scene"},
        {"id": 401, "text": "a bus terminal with several parked buses and lots of people walking in front of them", "feature": "Scene Type/Setting", "signals": "bus terminal", "confidence": "High", "notes": "Public transportation hub"},
        {"id": 426, "text": "a city street at night with a lit up kfc sign", "feature": "Scene Type/Setting", "signals": "city street at night", "confidence": "High", "notes": "Urban commercial nighttime scene"},
        {"id": 442, "text": "a kitchen is decorated in earth tones and is brightly lit", "feature": "Scene Type/Setting", "signals": "decorated kitchen", "confidence": "High", "notes": "Styled domestic cooking space"},
        {"id": 495, "text": "a family room with couch, tv, and desk and a wood floor", "feature": "Scene Type/Setting", "signals": "family room", "confidence": "High", "notes": "Multi-purpose domestic space"},
        {"id": 518, "text": "benches beneath an umbrella sit among palm trees on the water across from a large city", "feature": "Scene Type/Setting", "signals": "waterfront with city view", "confidence": "High", "notes": "Coastal recreational scene with urban backdrop"},
        {"id": 593, "text": "a bus is driving down a snowy road near the days inn and suites", "feature": "Scene Type/Setting", "signals": "snowy road, hotel area", "confidence": "High", "notes": "Winter commercial district"},
        {"id": 708, "text": "a black and white photo of a train yard in a large city", "feature": "Scene Type/Setting", "signals": "train yard, large city", "confidence": "High", "notes": "Industrial urban transportation scene"},
        {"id": 939, "text": "outdoor patio and front of a bar that serves hot dogs", "feature": "Scene Type/Setting", "signals": "outdoor patio, bar front", "confidence": "High", "notes": "Commercial food service exterior"}
    ];

humanRoleSamples = [
        {"id": 3, "text": "black and white photo of officer offering an item to bear sitting down", "feature": "Human Roles", "signals": "officer", "confidence": "High", "notes": "Law enforcement professional role"},
        {"id": 9, "text": "a man in business attire poses for a photo in an office building", "feature": "Human Roles", "signals": "business attire, office building", "confidence": "High", "notes": "Business professional context"},
        {"id": 68, "text": "a group of military people cutting a giant cake at a ceremony", "feature": "Human Roles", "signals": "military people", "confidence": "High", "notes": "Military service members in ceremonial context"},
        {"id": 100, "text": "a young man with a green hat is taking a selfie of himself in the public restroom", "feature": "Human Roles", "signals": "young man", "confidence": "Medium", "notes": "Age-based role identification"},
        {"id": 124, "text": "a young man with short, spiked-up hair and a towel over his shoulder shaves as a man behind him in a white hat adjusts a tie", "feature": "Human Roles", "signals": "man shaving, man adjusting tie", "confidence": "Medium", "notes": "Personal grooming roles and assistance"},
        {"id": 143, "text": "a baseball player stands at home plate with his bat in a yankees jersey", "feature": "Human Roles", "signals": "baseball player", "confidence": "High", "notes": "Professional athlete role"},
        {"id": 162, "text": "a man filming a women holding a microphone on a street corner", "feature": "Human Roles", "signals": "man filming, women with microphone", "confidence": "High", "notes": "Media production roles"},
        {"id": 212, "text": "one person is standing at a food cart ran by a man with a beard", "feature": "Human Roles", "signals": "food cart operator", "confidence": "High", "notes": "Food service vendor role"},
        {"id": 262, "text": "woman holding the reins of a horse in front of a crowd", "feature": "Human Roles", "signals": "woman with horse, crowd", "confidence": "Medium", "notes": "Equestrian handler with audience"},
        {"id": 287, "text": "a teenager with blue grey hair is wearing a white and blue shirt with a red tie", "feature": "Human Roles", "signals": "teenager", "confidence": "High", "notes": "Age-based role with formal dress"},
        {"id": 302, "text": "a woman in a top hat sits on a horse", "feature": "Human Roles", "signals": "woman, equestrian", "confidence": "Medium", "notes": "Rider role with formal attire"},
        {"id": 415, "text": "man with two parrots on his finger and one parrot on his shoulder", "feature": "Human Roles", "signals": "man with parrots", "confidence": "Medium", "notes": "Animal handler or trainer role"},
        {"id": 427, "text": "a man preparing food on top of a counter in a kitchen", "feature": "Human Roles", "signals": "man preparing food", "confidence": "High", "notes": "Cook or chef role"},
        {"id": 484, "text": "a man carrying a backpack getting on a green and white tour bus", "feature": "Human Roles", "signals": "man with backpack, tour bus", "confidence": "Medium", "notes": "Tourist or traveler role"},
        {"id": 510, "text": "a woman in a bow tie uniform, with a light shining behind her", "feature": "Human Roles", "signals": "woman in uniform", "confidence": "High", "notes": "Service professional in uniform"},
        {"id": 525, "text": "a man in a kilt with bagpipes leans against the window sill and talks on his cellphone", "feature": "Human Roles", "signals": "man with bagpipes, kilt", "confidence": "High", "notes": "Musician in traditional costume"},
        {"id": 610, "text": "a lady is cutting grapes from the stems at a market", "feature": "Human Roles", "signals": "lady cutting grapes, market", "confidence": "High", "notes": "Market vendor or worker"},
        {"id": 687, "text": "a boy in a purple baseball uniform is in the outfield", "feature": "Human Roles", "signals": "boy in baseball uniform", "confidence": "High", "notes": "Youth athlete role"},
        {"id": 759, "text": "a man with red hair, wearing a vest with a shirt and tie, making a drink at a bar area", "feature": "Human Roles", "signals": "man making drink, bar", "confidence": "High", "notes": "Bartender role"},
        {"id": 981, "text": "a jockey in a horse racing competition jumping a hurdle with his horse", "feature": "Human Roles", "signals": "jockey", "confidence": "High", "notes": "Professional equestrian athlete"}
    ];


quantitySamples = [
        {"id": 7, "text": "three women are actively playing a video game in front of a crowd", "feature": "Quantity", "signals": "three women", "confidence": "High", "notes": "Exact count of people"},
        {"id": 8, "text": "a bed with two pillows and a woman's legs with shoes laying on the bed", "feature": "Quantity", "signals": "two pillows", "confidence": "High", "notes": "Specific count of objects"},
        {"id": 24, "text": "a group of young women sanding next to each other holding tennis racquets", "feature": "Quantity", "signals": "group", "confidence": "Medium", "notes": "Indefinite quantity descriptor"},
        {"id": 27, "text": "several wine glasses have just a little bit of wine in each of them", "feature": "Quantity", "signals": "several, little bit", "confidence": "High", "notes": "Indefinite count with volume measure"},
        {"id": 32, "text": "three men on skis are flying in the air above a pool of blue water", "feature": "Quantity", "signals": "three men", "confidence": "High", "notes": "Exact count of people"},
        {"id": 46, "text": "two computer screens and keyboards side by side on a desktop", "feature": "Quantity", "signals": "two", "confidence": "High", "notes": "Specific count of paired items"},
        {"id": 56, "text": "a close up of two bags of luggage with a broken wheel", "feature": "Quantity", "signals": "two bags", "confidence": "High", "notes": "Exact count of luggage"},
        {"id": 88, "text": "three giraffes walk together across a field with trees behind them", "feature": "Quantity", "signals": "three giraffes", "confidence": "High", "notes": "Specific animal count"},
        {"id": 105, "text": "a pair of men doing tricks on skateboards off of rails", "feature": "Quantity", "signals": "pair", "confidence": "High", "notes": "Quantity expression for two"},
        {"id": 130, "text": "a group of people with wine posing for a photograph at a table", "feature": "Quantity", "signals": "group", "confidence": "Medium", "notes": "Collective quantity"},
        {"id": 159, "text": "a group of 5 police motor bikes lined in a row", "feature": "Quantity", "signals": "5", "confidence": "High", "notes": "Exact numerical count"},
        {"id": 177, "text": "the three giraffes are headed away from the camera", "feature": "Quantity", "signals": "three giraffes", "confidence": "High", "notes": "Specific animal count"},
        {"id": 207, "text": "a box of doughnuts some with swirls, some with cream cheese", "feature": "Quantity", "signals": "some", "confidence": "Medium", "notes": "Partitive quantity expressions"},
        {"id": 223, "text": "a group of people riding skis across a snow covered ground", "feature": "Quantity", "signals": "group", "confidence": "Medium", "notes": "Collective descriptor"},
        {"id": 240, "text": "two trains are parked alongside one another as if readying for a race", "feature": "Quantity", "signals": "two trains", "confidence": "High", "notes": "Specific count with comparison"},
        {"id": 247, "text": "four photos of a lady in different poses with a tennis racket", "feature": "Quantity", "signals": "four photos", "confidence": "High", "notes": "Exact count of images"},
        {"id": 292, "text": "two jets fly in formation against a white background", "feature": "Quantity", "signals": "two jets", "confidence": "High", "notes": "Specific aircraft count"},
        {"id": 331, "text": "three pancakes with butter on a yellow colored oval plate", "feature": "Quantity", "signals": "three pancakes", "confidence": "High", "notes": "Food item count"},
        {"id": 435, "text": "three dogs lying on a couch by a window, with a pumpkin in front", "feature": "Quantity", "signals": "three dogs", "confidence": "High", "notes": "Animal count"},
        {"id": 746, "text": "two glass vases, one containing one flower and the other containing two", "feature": "Quantity", "signals": "two vases, one flower, two", "confidence": "High", "notes": "Multiple quantity expressions"}
    ];


emotionSamples = [
        {"id": 14, "text": "a small giraffe studies a steep hill that borders his enclosure", "feature": "Emotion/Sentiment", "signals": "studies (curiosity)", "confidence": "Medium", "notes": "Implied cognitive/emotional state"},
        {"id": 35, "text": "a man standing near another man with his arm up in a room", "feature": "Emotion/Sentiment", "signals": "celebratory gesture", "confidence": "Medium", "notes": "Positive gestural expression"},
        {"id": 86, "text": "a tennis player poses for a picture in a black and white picture", "feature": "Emotion/Sentiment", "signals": "poses (confident)", "confidence": "Medium", "notes": "Positive self-presentation"},
        {"id": 99, "text": "a cute little brown teddy bear sits on a rock by a bush", "feature": "Emotion/Sentiment", "signals": "cute", "confidence": "Medium", "notes": "Positive aesthetic evaluation"},
        {"id": 140, "text": "a man is on his knees in anguish in front of a laptop", "feature": "Emotion/Sentiment", "signals": "anguish", "confidence": "High", "notes": "Explicit negative emotional state"},
        {"id": 148, "text": "a guy catches a frisbee as another looks on during a frisbee tournament beneath cloudy skies", "feature": "Emotion/Sentiment", "signals": "tournament (competitive excitement)", "confidence": "Medium", "notes": "Positive competitive context"},
        {"id": 158, "text": "a teddy bear sitting on the ground, looking at the camera", "feature": "Emotion/Sentiment", "signals": "looking at camera (engaging)", "confidence": "Medium", "notes": "Anthropomorphized engagement"},
        {"id": 185, "text": "a picture of a young man smiling while sitting in front of a wine glass", "feature": "Emotion/Sentiment", "signals": "smiling", "confidence": "High", "notes": "Clear positive facial expression"},
        {"id": 304, "text": "individuals are there commending and having a ton of fun of their life", "feature": "Emotion/Sentiment", "signals": "commending, having fun", "confidence": "High", "notes": "Explicit positive emotional expressions"},
        {"id": 330, "text": "a party of four standing at a tennis net one man is wearing a costume", "feature": "Emotion/Sentiment", "signals": "party, costume (playful)", "confidence": "Medium", "notes": "Festive and playful context"},
        {"id": 342, "text": "a bear looks on from behind the trunk of a birch tree", "feature": "Emotion/Sentiment", "signals": "looks on (watchful)", "confidence": "Medium", "notes": "Cautious or curious observation"},
        {"id": 411, "text": "a boy sitting on a yellow fire hydrant on a street side", "feature": "Emotion/Sentiment", "signals": "playful sitting", "confidence": "Medium", "notes": "Casual, relaxed behavior"},
        {"id": 417, "text": "a young child eating a banana with it's eyes open", "feature": "Emotion/Sentiment", "signals": "eating enjoyment", "confidence": "Medium", "notes": "Positive consumption behavior"},
        {"id": 544, "text": "a man wearing underclothes and an agonized expression, clutches his lower regions, while holding aloft a pair of scissors", "feature": "Emotion/Sentiment", "signals": "agonized expression", "confidence": "High", "notes": "Explicit negative emotional display"},
        {"id": 569, "text": "it is always fun to have a good friend along for the ride", "feature": "Emotion/Sentiment", "signals": "fun, good friend", "confidence": "High", "notes": "Positive social sentiment"},
        {"id": 600, "text": "here are is a stuffed toy dog and its beloved owner dressed in an animal costume", "feature": "Emotion/Sentiment", "signals": "beloved", "confidence": "High", "notes": "Explicit positive emotional attachment"},
        {"id": 711, "text": "a woman blowing out candles in a cake with a glass container", "feature": "Emotion/Sentiment", "signals": "blowing out candles (celebratory)", "confidence": "Medium", "notes": "Birthday celebration context"},
        {"id": 754, "text": "two men and a women are seated and smiling while a pizza sits in its carton on a table", "feature": "Emotion/Sentiment", "signals": "smiling", "confidence": "High", "notes": "Clear positive facial expressions"},
        {"id": 770, "text": "a happy woman in a white dress points at donuts in a display case", "feature": "Emotion/Sentiment", "signals": "happy", "confidence": "High", "notes": "Explicit positive emotional state"},
        {"id": 799, "text": "people sit smiling around a long table in the restaurant", "feature": "Emotion/Sentiment", "signals": "smiling", "confidence": "High", "notes": "Group positive expression"}
    ];
    
semanticSamples = [
        {"id": 5, "text": "the man in black outfit snowboards in the icy area", "feature": "Semantic Roles", "signals": "agent: man, action: snowboards, location: icy area", "confidence": "High", "notes": "Clear agent-action-location structure"},
        {"id": 29, "text": "a man and a boy are on a beach flying a kite", "feature": "Semantic Roles", "signals": "agents: man and boy, action: flying, patient: kite, location: beach", "confidence": "High", "notes": "Multiple agents with shared action"},
        {"id": 43, "text": "two people who are sitting on the back of an elephant", "feature": "Semantic Roles", "signals": "agents: two people, action: sitting, location: back of elephant", "confidence": "High", "notes": "Agent-action-locative relationship"},
        {"id": 72, "text": "a person preparing broccoli in a kitchen on top of a stove", "feature": "Semantic Roles", "signals": "agent: person, action: preparing, patient: broccoli, location: kitchen, instrument: stove", "confidence": "High", "notes": "Complex semantic structure with instrument"},
        {"id": 101, "text": "a man is riding a skateboard over a ramp while wearing a helmet", "feature": "Semantic Roles", "signals": "agent: man, action: riding, instrument: skateboard, path: over ramp", "confidence": "High", "notes": "Agent-instrument-path with manner"},
        {"id": 168, "text": "a man pulling two cows by ropes with a lot of people gathered together", "feature": "Semantic Roles", "signals": "agent: man, action: pulling, patient: cows, instrument: ropes", "confidence": "High", "notes": "Force dynamics with instrument"},
        {"id": 188, "text": "a man who is on a water ski being pulled by a boat", "feature": "Semantic Roles", "signals": "experiencer: man, action: being pulled, agent: boat, instrument: water ski", "confidence": "High", "notes": "Passive construction with causative agent"},
        {"id": 218, "text": "a young woman balances on a surf board as a man swims carries his board through the water behind her", "feature": "Semantic Roles", "signals": "agent1: woman, action1: balances, instrument: surf board; agent2: man, action2: carries", "confidence": "High", "notes": "Multiple semantic frames"},
        {"id": 230, "text": "a man prepares to hit a tennis ball across the net", "feature": "Semantic Roles", "signals": "agent: man, action: hit, patient: tennis ball, goal: across net", "confidence": "High", "notes": "Intentional action with directional goal"},
        {"id": 262, "text": "woman holding the reins of a horse in front of a crowd", "feature": "Semantic Roles", "signals": "agent: woman, action: holding, patient: reins, possessed: horse", "confidence": "High", "notes": "Control relationship through instrument"},
        {"id": 285, "text": "two people on a motorcycle, driving on a highway in the mountains", "feature": "Semantic Roles", "signals": "agents: two people, action: driving, instrument: motorcycle, location: highway, setting: mountains", "confidence": "High", "notes": "Shared agency with complex locative"},
        {"id": 394, "text": "an elephant being examined by two individuals next to a road", "feature": "Semantic Roles", "signals": "patient: elephant, action: being examined, agents: two individuals, location: next to road", "confidence": "High", "notes": "Passive patient with multiple agents"},
        {"id": 457, "text": "a man holding a large wooden spatula with bread on top of it", "feature": "Semantic Roles", "signals": "agent: man, action: holding, instrument: spatula, patient: bread", "confidence": "High", "notes": "Tool use with supported object"},
        {"id": 490, "text": "a man riding a horse next to a woman in a forest", "feature": "Semantic Roles", "signals": "agent: man, action: riding, instrument: horse, accompanier: woman, location: forest", "confidence": "High", "notes": "Social accompaniment in action"},
        {"id": 536, "text": "a woman on a bridge watches a truck enter the river", "feature": "Semantic Roles", "signals": "experiencer: woman, action: watches, patient: truck entering river, location: bridge", "confidence": "High", "notes": "Perception verb with complex patient"},
        {"id": 556, "text": "a woman mixes ingredients into a large pot on the stove", "feature": "Semantic Roles", "signals": "agent: woman, action: mixes, patient: ingredients, goal: into pot, location: stove", "confidence": "High", "notes": "Transfer action with directional goal"},
        {"id": 591, "text": "a man and children are riding on the back of an elephant", "feature": "Semantic Roles", "signals": "agents: man and children, action: riding, location: back of elephant", "confidence": "High", "notes": "Multiple agents with locative"},
        {"id": 713, "text": "a cow appears to run while two men on horses wearing hats are seen with lassos", "feature": "Semantic Roles", "signals": "agent1: cow, action1: run; agents2: men, instrument: horses, tool: lassos", "confidence": "High", "notes": "Multiple semantic frames with tools"},
        {"id": 756, "text": "a man cleaning his teeth with an electric toothbrush in a bathroom mirror", "feature": "Semantic Roles", "signals": "agent: man, action: cleaning, patient: teeth, instrument: toothbrush, location: bathroom", "confidence": "High", "notes": "Self-directed action with instrument"},
        {"id": 869, "text": "volunteer firemen with their truck shooting out a jet of water", "feature": "Semantic Roles", "signals": "agents: firemen, instrument: truck, action: shooting, patient: jet of water", "confidence": "High", "notes": "Professional action with equipment"}
    ];

spatialSamples = [
        {"id": 1, "text": "a cat with its paws on a computer mouse at a desk", "feature": "Spatial Relations", "signals": "on (contact), at (location)", "confidence": "High", "notes": "Contact and general location relations"},
        {"id": 6, "text": "a group of zebra walking on top of a lush green field", "feature": "Spatial Relations", "signals": "on top of", "confidence": "High", "notes": "Surface contact relation"},
        {"id": 18, "text": "a bird perched on top of a banana tree with lots of leaves", "feature": "Spatial Relations", "signals": "on top of", "confidence": "High", "notes": "Elevated position relation"},
        {"id": 38, "text": "some bears holding a british flag are in a trailer", "feature": "Spatial Relations", "signals": "in (containment)", "confidence": "High", "notes": "Containment spatial relation"},
        {"id": 51, "text": "a cat that is standing in the grass near a bird", "feature": "Spatial Relations", "signals": "in (location), near (proximity)", "confidence": "High", "notes": "Location and proximity relations"},
        {"id": 62, "text": "a bed sitting up against a gray wall next to a window", "feature": "Spatial Relations", "signals": "against (contact), next to (adjacency)", "confidence": "High", "notes": "Contact and adjacency relations"},
        {"id": 66, "text": "a blue chair sitting in a yard full of snow covered in snow", "feature": "Spatial Relations", "signals": "in (location), covered in (surface relation)", "confidence": "High", "notes": "Location and surface coverage"},
        {"id": 82, "text": "a little girl standing on top of a bench next to another little girl", "feature": "Spatial Relations", "signals": "on top of, next to", "confidence": "High", "notes": "Elevation and proximity"},
        {"id": 110, "text": "a lot of snow on the ground and a parking meter is sticking out of it", "feature": "Spatial Relations", "signals": "on (surface), sticking out of (partial containment)", "confidence": "High", "notes": "Surface and partial embedding"},
        {"id": 123, "text": "the couple are sitting by the bench on the boardwalk by the water", "feature": "Spatial Relations", "signals": "by (proximity), on (surface)", "confidence": "High", "notes": "Multiple proximity relations"},
        {"id": 145, "text": "a close up of a cat sitting on a couch in a living room", "feature": "Spatial Relations", "signals": "on (surface), in (containment)", "confidence": "High", "notes": "Surface contact and room containment"},
        {"id": 153, "text": "a bird sits atop a post near a boat dock with large boats in the background", "feature": "Spatial Relations", "signals": "atop, near, in background", "confidence": "High", "notes": "Elevation, proximity, and depth relations"},
        {"id": 170, "text": "a bench next to a tree with trees in the background", "feature": "Spatial Relations", "signals": "next to, in background", "confidence": "High", "notes": "Adjacency and depth relations"},
        {"id": 183, "text": "a flooded park bench and light pole behind a cluster of trees", "feature": "Spatial Relations", "signals": "behind", "confidence": "High", "notes": "Occlusion/depth relation"},
        {"id": 222, "text": "a white bench has shadows on it and is next to a door", "feature": "Spatial Relations", "signals": "on (surface), next to", "confidence": "High", "notes": "Surface projection and adjacency"},
        {"id": 264, "text": "a white slat bench bolted to tiled sidewalk, with the bench's middle seat slat spiral curled up to one end", "feature": "Spatial Relations", "signals": "bolted to, up to", "confidence": "High", "notes": "Attachment and directional relations"},
        {"id": 274, "text": "a herd of zebra standing on a strip of land between two rivers", "feature": "Spatial Relations", "signals": "on, between", "confidence": "High", "notes": "Surface and between relations"},
        {"id": 377, "text": "an exterior view with a double height train on the left, several bicycles parked in the middle and a bus on the right", "feature": "Spatial Relations", "signals": "on the left, in the middle, on the right", "confidence": "High", "notes": "Relative horizontal positioning"},
        {"id": 518, "text": "benches beneath an umbrella sit among palm trees on the water across from a large city", "feature": "Spatial Relations", "signals": "beneath, among, on, across from", "confidence": "High", "notes": "Multiple complex spatial relations"},
        {"id": 951, "text": "a cat has arranged itself mostly on a rug, but with its front paws in a shoe, as it rests on the shoe's mate and spreads its tail under a wooden stool", "feature": "Spatial Relations", "signals": "on, in, under", "confidence": "High", "notes": "Complex multi-surface positioning"}
    ];


all_feats = [objectSamples, actionSamples, sceneSamples, humanRoleSamples, quantitySamples, emotionSamples, semanticSamples, spatialSamples]
joined_feats =  []
for feat in all_feats:
    joined_feats += feat

feat_df = pd.DataFrame(joined_feats)
print(feat_df)
print(feat_df["feature"].value_counts())
print(feat_df["confidence"].value_counts())
print(feat_df[["feature", "confidence"]].value_counts())


main_df = pd.read_csv("./content/drive/MyDrive/base-clip-data/Cross-lingual-Test-Dataset-XTD10/merged_data.csv")
joined_df = pd.merge(feat_df, main_df, left_on='id', right_index=True, how='inner')

#joined_df = joined_df[joined_df['text'] == joined_df['XTD10_captions_en']]
print(joined_df[['text', 'XTD10_captions_en']])
print(len(joined_df["id"].unique()))

save_p = os.path.join("Synthetic_Multilingual_Eval_Dataset/", "XTD10_analysis_subset.csv")
joined_df.to_csv(save_p, index=False)
print(f"Saved at {save_p}")

