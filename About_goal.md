FlightRank 2025: Aeroclub RecSys Cup

Personalized Flight Recommendations for Business Travelers





Overview

Welcome aboard! ✈️

Imagine you're a business traveler searching for flights. You see dozens or even thousands of options with different prices, airlines, departure times, and durations. What makes you click "Book Now" on one specific flight? This competition challenges you to decode those preferences and build a recommendation system that can predict business traveler choices.

Competition Goal

Build an intelligent flights ranking model that predicts which flight option a business traveler will choose from search results.

Start

a month ago

Close

19 days to go

Description

Business travel presents unique challenges for recommendation systems. Unlike leisure travelers who prioritize cost or vacation preferences, business travelers must balance multiple competing factors: corporate travel policies, meeting schedules, expense compliance, and personal convenience. This creates complex decision patterns that are difficult to predict.

This competition challenges participants to solve a group-wise ranking problem where your models must rank flight options within each user search session. Each search session (ranker_id) represents a real user query with multiple flight alternatives, but only one chosen option. Your task is to build a model that can rank these options and identify the most likely selection.

The Challenge

The dataset contains real flight search sessions with various attributes including pricing, timing, route information, user features and booking policies. The key technical challenge lies in ranking flight options to identify the most suitable choices for each business traveler on their specific route and circumstances. This becomes particularly complex as the number of available options can vary dramatically - from a handful of alternatives on smaller routes to thousands of possibilities on major trunk routes. Your model must effectively rank this entire spectrum of options to enhance the user experience by accurately identifying which flights best match traveler preferences.

Why This Matters

Flight recommendation systems power major travel platforms serving millions of business travelers. Accurate ranking models can significantly improve user experience by surfacing relevant options faster, ultimately leading to higher conversion rates and customer satisfaction.

Your model will be evaluated based on ranking quality - how well it places the actually selected flight at the top of each search session's ranked list.

Evaluation

HitRate@3

Competition metric HitRate@3 measures the fraction of search sessions where the correct flight appears in your top-3 predictions.





Where:

|Q| is the number of search queries (unique ranker_id values)

rank_i is the rank position you assigned to the correct flight in query i

𝟙(rank_i ≤ 3) is 1 if the correct flight is in top-3, 0 otherwise

Example: If the correct flight is ranked 1st, 2nd, or 3rd, you get 1.0 points. Otherwise, you get 0 points.

Score range: 0 to 1, where 1 means the correct flight is always in top-3

Important Note on Group Size Filtering

The metric evaluation will only consider groups (ranker_id) with more than 10 flight options. Groups with 10 or fewer options are excluded from the final score calculation to focus on more challenging ranking scenarios where distinguishing between options is meaningful.

However, we have intentionally kept these smaller groups in both the training and test datasets because:

They represent real-world search scenarios

They provide additional training signal for your models

They help capture the full diversity of user behavior patterns

Submission Format

Training Data Target

In the training data, the selected column is binary:

1 = This flight was chosen by the traveler

0 = This flight was not chosen

Important: There is exactly one row with selected=1 per user search request (ranker_id). Each row within a ranker_id group represents a different flight option returned by the search system for that specific route and date.

Training data example:

Id,ranker_id,selected100,abc123,0 # Flight option 1 - not chosen101,abc123,0 # Flight option 2 - not chosen 102,abc123,1, # Flight option 3 - SELECTED by user103,abc123,0 # Flight option 4 - not chosen

Submission Format

Your submission must contain ranks (not probabilities) for each flight option:

Id,ranker_id,selected100,abc123,4101,abc123,2102,abc123,1103,abc123,3

Where:

Id matches the row identifier from the test set

ranker_id is the search session identifier (same as in test.csv)

selected is the rank you assign (1 = best option, 2 = second best, etc.)

Important: Maintain the exact same row order as in test.csv

In this example, your model predicts that:

Row 102 (Id=102) is the best option → Rank 1

Row 101 (Id=101) is second best → Rank 2

Row 103 (Id=103) is third best → Rank 3

Row 100 (Id=100) is the worst option → Rank 4

Submission Requirements

Preserve row order: Maintain the exact same row order as in test.csv

Complete rankings: For each user search request, you must rank ALL flight options returned by the search system

Valid permutation: Ranks within each ranker_id must be a valid permutation (1, 2, 3, …, N) where N is the number of rows in that group

No duplicate ranks: Each row within a ranker_id group must have a unique rank

Integer values: All ranks must be integers ≥ 1

Example for one user search request:

Training data shows:

ranker_id: abc123 → Row 102 was chosen (selected=1)



Your submission:

ranker_id: abc123

├── Row 100 → Rank 4 (worst option)

├── Row 101 → Rank 2 (second best)

├── Row 102 → Rank 1 (best - correctly predicted!)

└── Row 103 → Rank 3 (third best)

Validation

Your submission will be validated for:

Correct number of rows

Integer rank values

Valid rank permutations within each group

No duplicate ranks per search session

Basic anti-cheating measures

Note: The evaluation system expects you to transform your model's output (scores/probabilities) into ranks before submission. Higher model scores should correspond to lower rank numbers (1 = best).

Prizes

TOTAL PRIZE FUND: $10,000

Leaderboard Prizes:

1st Place: $2,500 or $5,000 (with bonus)

2nd Place: $1,750 or $3,500 (with bonus)

3rd Place: $750 or $1,500 (with bonus)

Bonus Performance Threshold:

Winners who achieve HitRate@3 ≥ 0.7 receive Bonus - double their prize amount.





Dataset Description

Data Description

Overview

This dataset contains flight booking options for business travelers along with user preferences and company policies. The task is to predict user flight selection preferences.

Data Structure

The dataset is organized around flight search sessions, where each session (identified by ranker_id) contains multiple flight options that users can choose from.

Main Data

'train.parquet' - train data

'test.parquet' - test data

'sample_submission.parquet' - submission example

JSONs Raw Additional Data

'jsons_raw.tar.kaggle'* - Archived raw data in JSONs files (150K files, ~50gb). To use the file as a regular .gz archive you should manually change extension to '.gz'. Example jsons_raw.tar.kaggle -> jsons_raw.tar.gz

'jsons_structure.md' - JSONs raw data structure description

Column Descriptions

Identifiers and Metadata

Id - Unique identifier for each flight option

ranker_id - Group identifier for each search session (key grouping variable for ranking)

profileId - User identifier

companyID - Company identifier

User Information

sex - User gender

nationality - User nationality/citizenship

frequentFlyer - Frequent flyer program status

isVip - VIP status indicator

bySelf - Whether user books flights independently

isAccess3D - Binary marker for internal feature

Company Information

corporateTariffCode - Corporate tariff code for business travel policies

Search and Route Information

searchRoute - Flight route: single direction without "/" or round trip with "/"

requestDate - Date and time when search was performed

Pricing Information

totalPrice - Total ticket price

taxes - Taxes and fees component

Flight Timing and Duration

legs0_departureAt - Departure time for outbound flight

legs0_arrivalAt - Arrival time for outbound flight

legs0_duration - Duration of outbound flight

legs1_departureAt - Departure time for return flight

legs1_arrivalAt - Arrival time for return flight

legs1_duration - Duration of return flight

Flight Segments

Each flight leg (legs0/legs1) can consist of multiple segments (segments0-3) when there are connections. Each segment contains:

Geography and Route

legs*_segments*_departureFrom_airport_iata - Departure airport code

legs*_segments*_arrivalTo_airport_iata - Arrival airport code

legs*_segments*_arrivalTo_airport_city_iata - Arrival city code

Airline and Flight Details

legs*_segments*_marketingCarrier_code - Marketing airline code

legs*_segments*_operatingCarrier_code - Operating airline code (actual carrier)

legs*_segments*_aircraft_code - Aircraft type code

legs*_segments*_flightNumber - Flight number

legs*_segments*_duration - Segment duration

Service Characteristics

legs*_segments*_baggageAllowance_quantity - Baggage allowance: small numbers indicate piece count, large numbers indicate weight in kg

legs*_segments*_baggageAllowance_weightMeasurementType - Type of baggage measurement

legs*_segments*_cabinClass - Service class: 1.0 = economy, 2.0 = business, 4.0 = premium

legs*_segments*_seatsAvailable - Number of available seats

Cancellation and Exchange Rules

Rule 0 (Cancellation)

miniRules0_monetaryAmount - Monetary penalty for cancellation

miniRules0_percentage - Percentage penalty for cancellation

miniRules0_statusInfos - Cancellation rule status (0 = no cancellation allowed)

Rule 1 (Exchange)

miniRules1_monetaryAmount - Monetary penalty for exchange

miniRules1_percentage - Percentage penalty for exchange

miniRules1_statusInfos - Exchange rule status

Pricing Policy Information

pricingInfo_isAccessTP - Compliance with corporate Travel Policy

pricingInfo_passengerCount - Number of passengers

Target Variable

selected - In training data: binary variable (0 = not selected, 1 = selected). In submission: ranks within ranker_id groups

Important Notes

Each ranker_id group represents one search session with multiple flight options

In training data, exactly one flight option per ranker_id has selected = 1

The prediction task requires ranking flight options within each search session

Segment numbering goes from 0 to 3, with segment 0 always present and higher numbers representing additional connections

JSONs Raw Data Archive

The competition includes a json_raw_tar.gz archive containing the original raw data from which the train and test datasets were extracted. This archive contains 150,770 JSON files, where each filename corresponds to a ranker_id group. Participants are allowed to use this raw data for feature enrichment and engineering, but it is not obligatory and only an option.

Warning: The uncompressed archive requires more than 50GB of disk space.

'jsons_raw.tar.kaggle'* - Compressed JSONs raw data (150K files, ~50gb). To use the file as a regular .gz archive you should manually change extension to '.gz'. Example jsons_raw.tar.kaggle -> jsons_raw.tar.gz

'jsons_structure.md' - JSONs raw data structure description

Submission Format

Training Data Target

In the training data, the selected column is binary:

1 = This flight was chosen by the traveler

0 = This flight was not chosen

Important: There is exactly one row with selected=1 per user search request (ranker_id). Each row within a ranker_id group represents a different flight option returned by the search system for that specific route and date.

Training data example:

Id,ranker_id,selected100,abc123,0 # Flight option 1 - not chosen101,abc123,0 # Flight option 2 - not chosen 102,abc123,1, # Flight option 3 - SELECTED by user103,abc123,0 # Flight option 4 - not chosen

Submission Format

Your submission must contain ranks (not probabilities) for each flight option:

Id,ranker_id,selected100,abc123,4101,abc123,2102,abc123,1103,abc123,3

Where:

Id matches the row identifier from the test set

ranker_id is the search session identifier (same as in test.csv)

selected is the rank you assign (1 = best option, 2 = second best, etc.)

Important: Maintain the exact same row order as in test.csv

In this example, your model predicts that:

Row 102 (Id=102) is the best option → Rank 1

Row 101 (Id=101) is second best → Rank 2

Row 103 (Id=103) is third best → Rank 3

Row 100 (Id=100) is the worst option → Rank 4

Submission Requirements

Preserve row order: Maintain the exact same row order as in test.csv

Complete rankings: For each user search request, you must rank ALL flight options returned by the search system

Valid permutation: Ranks within each ranker_id must be a valid permutation (1, 2, 3, …, N) where N is the number of rows in that group

No duplicate ranks: Each row within a ranker_id group must have a unique rank

Integer values: All ranks must be integers ≥ 1

Example for one user search request:

Training data shows:

ranker_id: abc123 → Row 102 was chosen (selected=1)



Your submission:

ranker_id: abc123

├── Row 100 → Rank 4 (worst option)

├── Row 101 → Rank 2 (second best)

├── Row 102 → Rank 1 (best - correctly predicted!)

└── Row 103 → Rank 3 (third best)

Validation

Your submission will be validated for:

Correct number of rows

Integer rank values

Valid rank permutations within each group

No duplicate ranks per search session

Basic anti-cheating measures

Note: The evaluation system expects you to transform your model's output (scores/probabilities) into ranks before submission. Higher model scores should correspond to lower rank numbers (1 = best).



Competition Rules

ENTRY IN THIS COMPETITION CONSTITUTES YOUR ACCEPTANCE OF THESE OFFICIAL COMPETITION RULES.

The Competition named below is a skills-based competition to promote and further the field of data science. You must register via the Competition Website to enter. To enter the Competition, you must agree to these Official Competition Rules, which incorporate by reference the provisions and content of the Competition Website and any Specific Competition Rules herein (collectively, the "Rules"). Please read these Rules carefully before entry to ensure you understand and agree. You further agree that Submission in the Competition constitutes agreement to these Rules. You may not submit to the Competition and are not eligible to receive the prizes associated with this Competition unless you agree to these Rules. These Rules form a binding legal agreement between you and the Competition Sponsor with respect to the Competition. Your competition Submissions must conform to the requirements stated on the Competition Website. Your Submissions will be scored based on the evaluation metric described on the Competition Website. Subject to compliance with the Competition Rules, Prizes, if any, will be awarded to Participants with the best scores, based on the merits of the data science models submitted. See below for the complete Competition Rules.

COMPETITION RULES

TOTAL PRIZE FUND: $10,000

1st Place: $2,500 or $5,000 (with bonus)

2nd Place: $1,750 or $3,500 (with bonus)

3rd Place: $750 or $1,500 (with bonus)

Bonus - Performance Threshold:

Winners who achieve HitRate@3 ≥ 0.7 receive double their prize amount.

Technical Requirements

Inference Limitations: Final solutions must not use GPU for predictions (GPU usage for training is allowed).

Winner Selection

Additional Verification: Private leaderboard metrics are not final. Organizers will request code from top-10 participants for final evaluation.

New Data Testing: Organizers reserve the right to test candidate solutions on unpublished 2025 data.

1. COMPETITION-SPECIFIC TERMS

Competition Name

FlightRank 2025: Aeroclub RecSys Challenge

Competition Sponsor

AEROCLUB LTD

Competition Sponsor Address

Republic of Kazakhstan, Almaty, st. Varlamova 27a, 050005

Competition Website

https://www.kaggle.com/competitions/aeroclub-recsys-2025

Winner License Type

Non-Exclusive License (see section 3.3)

Data Access and Use

Competition Use Only (see section 3.1)

2.SPECIFIC RULES

2.1 Team Limits

a. The maximum team size is five (5).

b. Team mergers are allowed and can be performed by the team leader. In order to merge, the combined team must have a total submission count less than or equal to the maximum allowed as of the team merger deadline.

2.2 Submission Limits

a. You may submit a maximum of five (5) submissions per day.

b. You may select up to two (2) final submissions for judging.

2.3 Competition Timeline

Competition timeline dates (including entry deadline, final submission deadline, start date, and team merger deadline) are reflected on the competition's Overview > Timeline page.

3. Competition Data

3.1 Data Access and Use

You may access and use the competition data only for participating in the competition and on Kaggle.com forums. The competition sponsor reserves the right to disqualify any participant who uses the competition data other than as permitted by the competition website and these rules.

3.2 Data Security

You agree to use reasonable and suitable measures to prevent persons who have not formally agreed to these rules from gaining access to the competition data. You agree not to transmit, duplicate, publish, redistribute or otherwise provide or make available the competition data to any party not participating in the competition. You agree to notify Kaggle immediately upon learning of any possible unauthorized transmission of or unauthorized access to the competition data.

3.3 Winner License

As a condition to being awarded a prize, you hereby grant the competition sponsor the following license with respect to your submission if you are a competition winner:

Non-Exclusive License: You hereby grant and will grant to competition sponsor and its designees a worldwide, non-exclusive, sub-licensable, transferable, fully paid-up, royalty-free, perpetual, irrevocable right to use, reproduce, distribute, create derivative works of, publicly perform, publicly display, digitally perform, make, have made, sell, offer for sale and import your winning submission and the source code used to generate the submission, in any media now known or developed in the future, for any purpose whatsoever, commercial or otherwise, without further approval by or payment to you.

For generally commercially available software that you used to generate your submission that is not owned by you, but that can be procured by the competition sponsor without undue expense, you do not need to grant the license for that software.

4 External Data and Tools

4.1 External Data

You may use data other than the competition data ("external data") to develop and test your submissions. However, you will ensure the external data is either publicly available and equally accessible to use by all participants of the competition at no cost to the other participants.

4.2 Reasonableness Criteria

The use of external data and models is acceptable unless their use must be "reasonably accessible to all" and of "minimal cost". A small subscription charge to use additional elements of large language models is acceptable. Purchasing a license to use a proprietary dataset that exceeds the cost of a prize in the competition would not be considered reasonable.

5. Eligibility

Unless otherwise stated in the competition-specific rules above, employees, interns, contractors, officers and directors of the competition sponsor, Kaggle Inc., and their respective parent companies, subsidiaries and affiliates may enter and participate in the competition, but are not eligible to win any prizes.

6. Winner's Obligations

a. Delivery & Documentation

Final model must be implemented as an Estimator with a predict(X) method. As a condition of receipt of the prize, the prize winner must deliver the final model's software code as used to generate the winning submission and associated documentation (consistent with the winning model documentation template available on the Kaggle wiki at https://www.kaggle.com/WinningModelDocumentationGuidelines) to the competition sponsor. The delivered software code must be capable of generating the winning submission and contain a description of resources required to build and/or run the executable code successfully.

b. Detailed Description

You may be required to provide a detailed description of how the winning submission was generated. This may include a detailed description of methodology, where one must be able to reproduce the approach by reading the description, and includes a detailed explanation of the architecture, preprocessing, loss function, training details, hyper-parameters, etc. The description should also include a link to a code repository with complete and detailed instructions so that the results obtained can be reproduced.