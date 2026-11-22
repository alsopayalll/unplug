const API = "http://127.0.0.1:8000/predict";

const form = document.getElementById("inpForm");

const age = document.getElementById("age");
const gender = document.getElementById("gender");
const country = document.getElementById("countryInp");
const academic = document.getElementById("acadlevel");

const mostUsedPlatform = document.getElementById("mostusedplat");
const mainPurpose = document.getElementById("mainpurp");

const avgDailyHours = document.getElementById("avgDailyHours");
const unlocks = document.getElementById("unlocks");

const studyHours = document.getElementById("studyhours");
const physicalActivityHours = document.getElementById("physActiviyHours");
const sleepHours = document.getElementById("sleepHours");

const stressLevel = document.getElementById("stresslevel");

const submitButton = document.getElementById("formsubmitButton");

const resultBox = document.getElementById("result");


// Slider value display elements
const avgDailyHoursValue = document.getElementById("avgDailyHoursValue");
const studyHoursValue = document.getElementById("studyhoursValue");
const physicalActivityHoursValue = document.getElementById("physActiviyHoursValue");
const sleepHoursValue = document.getElementById("sleepHoursValue");

function updateSliderValue(slider, display, unit) {

    display.textContent = `${Number(slider.value).toFixed(1)} ${unit}`;

}

avgDailyHours.addEventListener("input", function () {

    updateSliderValue(avgDailyHours, avgDailyHoursValue, "hrs");

});

studyHours.addEventListener("input", function () {

    updateSliderValue(studyHours, studyHoursValue, "hours");

});

physicalActivityHours.addEventListener("input", function () {

    updateSliderValue(
        physicalActivityHours,
        physicalActivityHoursValue,
        "hours"
    );

});

sleepHours.addEventListener("input", function () {

    updateSliderValue(sleepHours, sleepHoursValue, "hours");

});

function normalizeCountry(value) {

    const countryMap = {
        "united states": "USA",
        "united states of america": "USA",
        "us": "USA",

        "united kingdom": "UK",
        "great britain": "UK",

        "india": "India",
        "canada": "Canada",
        "australia": "Australia",
        "germany": "Germany",
        "mexico": "Mexico",
        "turkey": "Turkey",
        "france": "France"
    };

    const cleanedCountry = value.trim();

    return countryMap[cleanedCountry.toLowerCase()] || cleanedCountry;

}

function showResult(score) {

    resultBox.replaceChildren();

    const heading = document.createElement("h2");
    heading.textContent = "Your Predicted Score";

    const scoreText = document.createElement("p");
    scoreText.textContent = Number(score).toFixed(2);

    const message = document.createElement("p");
    message.textContent =
        "This is a model-generated prediction based on the information you provided. It is not a medical diagnosis.";

    resultBox.appendChild(heading);
    resultBox.appendChild(scoreText);
    resultBox.appendChild(message);

    resultBox.scrollIntoView({
        behavior: "smooth",
        block: "center"
    });

}

function showError(message) {

    resultBox.replaceChildren();

    const errorHeading = document.createElement("h3");
    errorHeading.textContent = "Something went wrong";

    const errorText = document.createElement("p");
    errorText.textContent = message;

    resultBox.appendChild(errorHeading);
    resultBox.appendChild(errorText);

    resultBox.scrollIntoView({
        behavior: "smooth",
        block: "center"
    });

}

form.addEventListener("submit", async function (event) {

    // Prevent page refresh
    event.preventDefault();

    // Check browser form validation
    if (!form.reportValidity()) {
        return;
    }

    // Clear previous result
    resultBox.replaceChildren();

    submitButton.disabled = true;
    submitButton.textContent = "Calculating...";

    const inputData = {

        age: Number(age.value),

        gender: gender.value,

        country: normalizeCountry(country.value),

        academic_level: academic.value,

        most_used_platform: mostUsedPlatform.value,

        purpose_of_use: mainPurpose.value,

        avg_daily_usage_hours: Number(avgDailyHours.value),

        daily_unlocks: Number(unlocks.value),

        study_hours: Number(studyHours.value),

        physical_activity_hours: Number(physicalActivityHours.value),

        sleep_hours_per_night: Number(sleepHours.value),

        stress_level: stressLevel.value

    };


    console.log("Data being sent to backend:", inputData);

    try {

        const response = await fetch(API, {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify(inputData)

        });

        const data = await response.json();

        if (!response.ok) {

            let errorMessage = "Unable to generate prediction.";

            if (Array.isArray(data.detail)) {

                errorMessage = data.detail
                    .map(error => error.msg)
                    .join(", ");

            } else if (typeof data.detail === "string") {

                errorMessage = data.detail;

            }

            throw new Error(errorMessage);

        }

        console.log("Prediction received:", data);

        if (typeof data.predicted_score !== "number") {

            throw new Error("Invalid prediction received from backend.");

        }

        showResult(data.predicted_score);


    }

    catch (error) {

        console.error("Prediction error:", error);

        if (error instanceof TypeError) {

            showError(
                "Cannot connect to the backend. Please make sure your FastAPI server is running at http://127.0.0.1:8000."
            );

        } else {

            showError(error.message);

        }

    }

    finally {

        // Enable button again
        submitButton.disabled = false;
        submitButton.textContent = "Check my score";

    }

});
