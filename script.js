
const API_URL = "http://127.0.0.1:8000/recommend";

document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("inpForm");
    const resultDiv = document.getElementById("result");
    const submitButton = document.getElementById("formsubmitButton");

    // --------------------------------
    // SLIDER VALUE DISPLAY
    // --------------------------------

    const sliders = [
        {
            input: "avgDailyHours",
            display: "avgDailyHoursValue",
            suffix: " hrs"
        },
        {
            input: "studyhours",
            display: "studyhoursValue",
            suffix: " hours"
        },
        {
            input: "physActiviyHours",
            display: "physActiviyHoursValue",
            suffix: " hours"
        },
        {
            input: "sleepHours",
            display: "sleepHoursValue",
            suffix: " hours"
        }
    ];

    sliders.forEach((item) => {
        const input = document.getElementById(item.input);
        const display = document.getElementById(item.display);

        if (input && display) {
            const updateValue = () => {
                display.textContent =
                    Number(input.value).toFixed(1) + item.suffix;
            };

            updateValue();
            input.addEventListener("input", updateValue);
        }
    });

    // --------------------------------
    // COLLECT FORM DATA
    // --------------------------------

    function getFormData() {
        return {
            age: Number(document.getElementById("age").value),

            gender: document.getElementById("gender").value,

            country: document.getElementById("countryInp").value.trim(),

            academic_level: document.getElementById("acadlevel").value,

            most_used_platform: document.getElementById("mostusedplat").value,

            purpose_of_use: document.getElementById("mainpurp").value,

            avg_daily_usage_hours: Number(
                document.getElementById("avgDailyHours").value
            ),

            daily_unlocks: Number(
                document.getElementById("unlocks").value
            ),

            study_hours: Number(
                document.getElementById("studyhours").value
            ),

            physical_activity_hours: Number(
                document.getElementById("physActiviyHours").value
            ),

            sleep_hours_per_night: Number(
                document.getElementById("sleepHours").value
            ),

            stress_level: document.getElementById("stresslevel").value
        };
    }

    // --------------------------------
    // FORMAT FEATURE NAMES
    // --------------------------------

    function formatFeatureName(feature) {
        if (feature.startsWith("Grouped_Countries_")) {
            return "Country Group: " +
                feature.replace("Grouped_Countries_", "");
        }

        if (feature.startsWith("Purpose_Of_Use_")) {
            return "Purpose: " +
                feature.replace("Purpose_Of_Use_", "");
        }

        if (feature.startsWith("Most_Used_Platform_")) {
            return "Platform: " +
                feature.replace("Most_Used_Platform_", "");
        }

        if (feature.startsWith("Gender_")) {
            return "Gender: " +
                feature.replace("Gender_", "");
        }

        if (feature.startsWith("Academic_Level_")) {
            return "Academic Level: " +
                feature.replace("Academic_Level_", "");
        }

        return feature.replaceAll("_", " ");
    }

    // --------------------------------
    // FORMAT NUMBERS
    // --------------------------------

    function formatNumber(value) {
        return Number(value).toFixed(3);
    }

    // --------------------------------
    // PREVENT HTML INJECTION
    // --------------------------------

    function escapeHTML(value) {
        return String(value).replace(/[&<>"']/g, (character) => {
            const entities = {
                "&": "&amp;",
                "<": "&lt;",
                ">": "&gt;",
                '"': "&quot;",
                "'": "&#039;"
            };

            return entities[character];
        });
    }

    // --------------------------------
    // DISPLAY SHAP + GEMINI RESULTS
    // --------------------------------

    function displayResults(data) {
        const predictedScore = Number(data.predicted_score);
        const baseline = Number(data.baseline);
        const features = data.top_features;

        if (
            !Number.isFinite(predictedScore) ||
            !Number.isFinite(baseline) ||
            !Array.isArray(features)
        ) {
            throw new Error(
                "Invalid prediction or SHAP response received from backend."
            );
        }

        const difference = predictedScore - baseline;

        // --------------------------------
        // SHAP FEATURE BARS
        // --------------------------------

        const maxContribution = Math.max(
            ...features.map((feature) =>
                Math.abs(Number(feature.contribution) || 0)
            ),
            0.000001
        );

        const featureHTML = features.map((feature) => {
            const contribution = Number(feature.contribution);

            if (!Number.isFinite(contribution)) {
                throw new Error("Invalid SHAP contribution received.");
            }

            const isPositive = contribution >= 0;

            const contributionClass =
                isPositive ? "positive" : "negative";

            const barWidth =
                (Math.abs(contribution) / maxContribution) * 100;

            const sign = isPositive ? "+" : "";

            return `
                <div class="feature-row">

                    <div class="feature-info">

                        <span class="feature-name">
                            ${escapeHTML(formatFeatureName(feature.feature))}
                        </span>

                        <span class="feature-value ${contributionClass}">
                            ${sign}${formatNumber(contribution)}
                        </span>

                    </div>

                    <div class="feature-bar-track">

                        <div
                            class="feature-bar-fill ${contributionClass}"
                            style="width: ${barWidth}%">
                        </div>

                    </div>

                </div>
            `;
        }).join("");

        // --------------------------------
        // GEMINI RECOMMENDATIONS
        // --------------------------------

        const recommendations = Array.isArray(data.recommendations)
            ? data.recommendations
            : [];

        const recommendationSource =
            data.recommendation_source === "gemini"
                ? "Gemini AI"
                : "Fallback recommendations";

        const recommendationClass =
            data.recommendation_source === "gemini"
                ? "gemini"
                : "fallback";

        const recommendationHTML = recommendations.length > 0
            ? recommendations.map((recommendation) => `
                <li>
                    ${escapeHTML(recommendation)}
                </li>
            `).join("")
            : `
                <li>
                    No recommendations are available right now.
                    Please try again later.
                </li>
            `;

        const recommendationMessage = data.recommendation_message
            ? escapeHTML(data.recommendation_message)
            : "";

        // --------------------------------
        // COMPLETE RESULT HTML
        // --------------------------------

        resultDiv.innerHTML = `

            <div class="result-heading">

                <h2>Your <span>Unplug</span> Analysis</h2>

                <p>
                    Here's what our machine learning model found
                    from your responses.
                </p>

            </div>


            <div class="result-score">

                <div class="score-label">
                    Predicted Mental Health Score
                </div>

                <div class="score-number">
                    ${formatNumber(predictedScore)}
                </div>

                <div class="score-label">
                    Model prediction
                </div>

            </div>


            <div class="result-summary">

                <div class="summary-card">

                    <div class="summary-label">
                        Baseline Score
                    </div>

                    <div class="summary-value">
                        ${formatNumber(baseline)}
                    </div>

                </div>


                <div class="summary-card">

                    <div class="summary-label">
                        Difference from Baseline
                    </div>

                    <div class="summary-value">
                        ${difference >= 0 ? "+" : ""}
                        ${formatNumber(difference)}
                    </div>

                </div>

            </div>


            <div class="feature-section">

                <h3>What influenced your prediction?</h3>

                <p class="feature-description">
                    These are the top ${features.length} features
                    that contributed to your individual prediction.
                </p>

                <div class="feature-list">

                    ${featureHTML}

                </div>


                <div class="result-legend">

                    <div class="legend-item">

                        <span class="legend-dot positive"></span>

                        Pushes prediction higher

                    </div>


                    <div class="legend-item">

                        <span class="legend-dot negative"></span>

                        Pushes prediction lower

                    </div>

                </div>

            </div>


            <!-- PERSONALIZED RECOMMENDATIONS -->

            <div class="recommendation-section">

                <div class="recommendation-heading">

                    <h3>Your Personalized Recommendations</h3>

                    <span class="recommendation-source ${recommendationClass}">
                        ${escapeHTML(recommendationSource)}
                    </span>

                </div>

                <p class="recommendation-description">
                    Based on your responses and the factors
                    influencing your prediction, here are some
                    suggestions to help you maintain a balanced routine.
                </p>

                <ul class="recommendation-list">
                    ${recommendationHTML}
                </ul>

                <p class="recommendation-message">
                    ${recommendationMessage}
                </p>

            </div>


            <!-- EXPLANATION AND DISCLAIMER -->

            <div class="result-note">

                <strong>How to read this analysis:</strong>

                SHAP values explain how individual features influence
                the model's prediction compared with its baseline.
                Positive contributions push the prediction higher,
                while negative contributions push it lower.

                <br><br>

                <strong>Disclaimer:</strong>

                This is an AI-based prediction for educational purposes.
                It is not a medical diagnosis or a substitute for
                professional mental health support.

            </div>

        `;

        resultDiv.style.display = "block";

        resultDiv.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });
    }

    // --------------------------------
    // FORM SUBMISSION
    // --------------------------------

    if (form) {
        form.addEventListener("submit", async (event) => {
            event.preventDefault();

            resultDiv.style.display = "block";

            resultDiv.innerHTML = `
                <div class="loading-message">
                    <h3>Analyzing your responses...</h3>
                    <p>
                        Please wait while Unplug generates your
                        prediction, explanation and recommendations.
                    </p>
                </div>
            `;

            submitButton.disabled = true;
            submitButton.textContent = "Analyzing...";

            try {
                const formData = getFormData();

                console.log("Sending data:", formData);

                const response = await fetch(API_URL, {
                    method: "POST",

                    headers: {
                        "Content-Type": "application/json"
                    },

                    body: JSON.stringify(formData)
                });

                const data = await response.json();

                if (!response.ok) {
                    const errorMessage =
                        data.detail || "Unable to generate recommendations.";

                    throw new Error(
                        typeof errorMessage === "string"
                            ? errorMessage
                            : JSON.stringify(errorMessage)
                    );
                }

                console.log("Complete recommendation response:", data);

                displayResults(data);

            } catch (error) {
                console.error("Recommendation error:", error);

                resultDiv.style.display = "block";

                resultDiv.innerHTML = `
                    <div class="result-error">

                        <h3>Something went wrong!</h3>

                        <p>
                            ${escapeHTML(error.message)}
                        </p>

                        <p>
                            Please check whether your backend server
                            is running and try again.
                        </p>

                    </div>
                `;

                resultDiv.scrollIntoView({
                    behavior: "smooth",
                    block: "center"
                });

            } finally {
                submitButton.disabled = false;
                submitButton.textContent = "Get My Score";
            }
        });
    }
});
