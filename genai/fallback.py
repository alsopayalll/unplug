def generate_fallback_recommendations(user_data):

    recommendations = []

    usage = float(user_data.get("Avg_Daily_Usage_Hours", 0))
    unlocks = int(user_data.get("Daily_Unlocks", 0))
    sleep = float(user_data.get("Sleep_Hours_Per_Night", 0))
    study = float(user_data.get("Study_Hours", 0))
    activity = float(user_data.get("Physical_Activity_Hours", 0))
    stress = str(user_data.get("Stress_Level", "")).lower()

    # 1. Screen time suggestions
    if usage >= 6:
        recommendations.append(
            "Try setting specific time limits for recreational "
            "screen use and taking regular screen breaks."
        )
    elif usage >= 4:
        recommendations.append(
            "Maintain a balanced screen routine by taking "
            "short breaks between extended phone sessions."
        )
    else:
        recommendations.append(
            "Continue being mindful of your screen time "
            "and maintain a balanced digital routine."
        )

    # 2. Phone unlock suggestions
    if unlocks >= 80:
        recommendations.append(
            "Consider checking your phone at planned intervals "
            "and turning off non-essential notifications."
        )
    elif unlocks >= 40:
        recommendations.append(
            "Try grouping notifications together to reduce "
            "unnecessary interruptions."
        )

    # 3. Sleep suggestions
    if 0 < sleep < 7:
        recommendations.append(
            "Consider maintaining a consistent sleep schedule "
            "and reducing screen use before bedtime."
        )
    elif sleep >= 7:
        recommendations.append(
            "Continue maintaining a regular sleep routine."
        )

    # 4. Study routine suggestions
    if study >= 5:
        recommendations.append(
            "Break longer study sessions into focused intervals "
            "with short rest breaks in between."
        )

    # 5. Physical activity suggestions
    if activity < 1:
        recommendations.append(
            "Try including a short walk, stretching session, "
            "or other enjoyable physical activity in your day."
        )

    # 6. Stress management suggestions
    if stress in ["high", "very high"]:
        recommendations.append(
            "Consider taking short relaxation breaks, practising "
            "slow breathing, and reaching out to someone you trust "
            "if you feel overwhelmed."
        )
    elif stress == "medium":
        recommendations.append(
            "Make some time for relaxing activities between "
            "your study and daily responsibilities."
        )
    elif stress == "low":
        recommendations.append(
            "Continue making time for activities that help "
            "you relax and maintain balance."
        )

    # Keep a maximum of five suggestions
    recommendations = recommendations[:5]

    # Daily challenge
    daily_challenge = (
        "Take one 20-minute break from recreational screen use "
        "today and spend that time doing something you enjoy."
    )

    # Format output
    result = "PERSONALIZED DIGITAL WELLNESS SUGGESTIONS\n\n"

    for index, suggestion in enumerate(recommendations, start=1):
        result += f"{index}. {suggestion}\n\n"

    result += f"DAILY CHALLENGE:\n{daily_challenge}\n\n"

    result += (
        "These suggestions are general wellness guidance "
        "based on the information provided. They are not "
        "a medical assessment or diagnosis."
    )

    return result
