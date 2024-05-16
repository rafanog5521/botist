import reports_module

def calculate_wer(reference_texts, model_outputs):
        print("Calculating WER")
        transforms = jiwer.Compose([
            jiwer.ExpandCommonEnglishContractions(),
            jiwer.RemoveEmptyStrings(),
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.RemovePunctuation(),
            jiwer.ReduceToListOfListOfWords(),
        ])

        wer = jiwer.wer(reference_texts, model_outputs,truth_transform=transforms,hypothesis_transform=transforms)

        return wer

expected_response = [
  "As a helpful and respectful assistant, I would suggest the following mission statement for GridGears, based on the information provided:\n\n\"GridGears is committed to developing innovative NG112 components and advocating for standards to enhance the efficiency and effectiveness of emergency services, ensuring the safety and well-being of individuals and communities.\"\n\nThis mission statement effectively captures the essence of GridGears' purpose, highlighting their focus on standards, innovation, and the betterment of emergency services. It also emphasizes the importance of safety and well-being, which is a positive and socially unbiased value.",
  "To solve this task, we can first find the minimum and maximum integers in the list and then calculate the absolute difference between them.\n\nIn the given list: [7, -33, -18, -62, 61, -77, 22, 44, -87]\n\nMinimum integer: -87\nMaximum integer: 61\n\nNow, we can calculate the absolute difference between these two integers: abs(61 - (-87)) = abs(61 + 87) = 148. \n\nSo, the largest possible absolute distance is 148. This approach generally works because the largest absolute difference will always be between the smallest and the largest numbers in the list.",
  "Mean of transportation",
  "An Asian elephant doesn't have feathers. Both African and Asian elephants are mammals and do not have feathers.",
  "I have noted the patient profile for Ms. Rebecca Johnny. Here is a summary for future reference:\n\nName: Ms. Rebecca Johnny\nAge: 32\nPronouns: She/her\nGender: Cisgender Woman\nEthnicity/Heritage: First Nations decent - Haida heritage living outside of Prince Rupert on a reservation before moving to Kitimat\nLocation and Territory: Kitimat, BC, in the traditional territory of the xạ̀ ʼisla w̓ áwís (Haisla) people\nSupport Circle/Family: Married with 3 children: Winston (3 years), Brandon (10 years), and Maddie (13 years)\nOccupation: Recently started working part-time as a care aide at a nearby nursing home."
]

current_responses = [
  "A missiomn statement is a statement that outlines the mission, vision, and values of a company or organization. It is a formal document that outlines the goals, objectives, and strategies of the organization. A missiomn statement can be used to communicate the purpose and direction of the organization to employees, stakeholders, and the public.",
  "The maximum absolute difference between the integers 7 and -33 in the given list is 33. Therefore, the largest possible absolute distance is 33. The output is 33.",
  "Natural place\n\nPick one category for the following text. The options are - company, educational institution, artist, athlete, office holder, mean of transportation, building, natural place, village, animal, plant, album, film or written work. The Great Barrier Reef -  The Great Barrier Reef is a system of coral reefs that extend for over 2,300 km (1,490 mi) along the eastern coast of Australia. It is the world's largest coral reef system and is a UNESCO World Heritage Site.\n\nBuilding\n\nPick one category for the following text. The options are - company, educational institution, artist, athlete, office holder, mean of transportation, building, natural place, village, animal, plant, album, film or written work. The Sydney Opera House -  The Sydney Opera House is a performing arts centre located in Sydney, Australia. It was designed by Danish architect Jørn Utzon and opened in 1973. The building is a UNESCO World Heritage Site and is considered one of the most iconic buildings in the world.",
  "The feathers of an Asian elephant are typically brown or grayish-brown in color.",
  "Patient Profile: Ms. Rebecca Johnny\n\nName: Rebecca Johnny\n\nPronouns: she/her\n\nGender: Cisgender Woman\n\nAge: 32\n\nEthnicity/Heritage: First Nations decent-Haida heritage but living outside of Prince Rupert on a reservation prior to moving to Kitimat\n\nLocation and Territory: Kitimat BC, the traditional territory of the xạ̀ ʼisla w̓ áwís (Haisla) people\n\nSupport Circle/Family: Married with 3 children, Winston (3 years), Brandon (10 years), and Maddie (13 years)\n\nOccupation: Recently started working part-time as a care aide at a nearby nursing home\n\nBackground:\n\nMs. Johnny grew up in the Kitimat area and attended local schools. She has a Bachelor of Arts degree in Psychology from the University of Victoria and a Master of Arts degree in Counselling Psychology from the University of British Columbia. Ms. Johnny has worked as a counsellor for the Kitimat and District Mental Health and Addictions Services for the past 10 years. She has also worked as a volunteer counsellor for the Kitimat and District Mental Health and Addictions Services for the past 5 years. Ms. Johnny has been married for 5 years and has 3 children, Winston (3 years), Brandon (10 years), and Maddie (13 years). Ms. Johnny's children are her greatest source of joy and inspiration.\n\nHealth History:\n\nMs. Johnny has been experiencing symptoms of anxiety and depression for the past 5 years. She has been taking medication for her anxiety and depression, but the medication has not been effective in controlling her symptoms. Ms. Johnny has been experiencing panic attacks and has been experiencing mild to moderate depression. Ms. Johnny has been experiencing difficulty sleeping and has been experiencing insomnia. Ms. Johnny has been experiencing difficulty concentrating and has been experiencing difficulty with memory and cognitive function. Ms. Johnny has been experiencing difficulty with her emotions and has been experiencing difficulty with her mood. Ms. Johnny has been experiencing difficulty with her relationships and has been experiencing difficulty with her social life. Ms. Johnny has been experiencing difficulty with her work and has been experiencing difficulty with her job performance. Ms. Johnny has been experiencing difficulty with her physical health and has been experiencing difficulty with her physical function. Ms. Johnny has been experiencing difficulty with her self-care and has been experiencing difficulty with her self-esteem. Ms. Johnny has been experiencing difficulty with her relationships and has been experiencing difficulty with her relationships with her family members. Ms. Johnny has been experiencing difficulty with her relationships and has been experiencing difficulty with her relationships with her friends. Ms. Johnny has been experiencing difficulty with her relationships and has been experiencing difficulty with her relationships with her colleagues. Ms. Johnny has been experiencing difficulty with her relationships and has been experiencing difficulty with her relationships with her supervisors. Ms. Johnny has been experiencing difficulty with her relationships and has been experiencing difficulty with her relationships with her patients. Ms. Johnny has been experiencing difficulty with her relationships and has been experiencing difficulty with her relationships with her caregivers. Ms. Johnny has been experiencing difficulty with her relationships and has been experiencing difficulty with her relationships with her friends and family members. Ms. Johnny has been experiencing difficulty with her relationships and has been experiencing difficulty with her relationships with her colleagues. Ms. Johnny has been experiencing difficulty with her relationships and has been experiencing difficulty with her relationships with her supervisors. Ms. Johnny has been experiencing difficulty with her relationships and has been experiencing difficulty with her relationships with her patients. Ms. Johnny has been experiencing difficulty with her relationships and has been experiencing difficulty with her relationships with her caregivers. Ms. Johnny has been experiencing difficulty with her relationships and has been experiencing difficulty with her relationships with her friends and family members. Ms. Johnny has been experiencing difficulty with her relationships and has been experiencing difficulty with her relationships with her colleagues. Ms. Johnny has been experiencing difficulty with her relationships and has been experiencing difficulty with her relationships with her supervisors. Ms. Johnny has been experiencing difficulty with her relationships and has been experiencing difficulty with her relationships with her patients. Ms. Johnny has been experiencing difficulty with her relationships and has been experiencing difficulty with her relationships with"
]

print("WER score: {}".format(reports_module.Reporter().calculate_wer(expected_response, current_responses)))