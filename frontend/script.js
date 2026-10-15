
const API_URL = "http://127.0.0.1:5000";

// HTML elements
const searchForm = document.getElementById("searchForm");
const searchInput = document.getElementById("searchInput");
const searchButton = document.getElementById("searchButton");
const recipeResults = document.getElementById("recipeResults");
const message = document.getElementById("message");
const resultsTitle = document.getElementById("resultsTitle");
const resultsInfo = document.getElementById("resultsInfo");
const resultCount = document.getElementById("resultCount");

// Keep only the latest search's AI response
let currentSearchId = 0;

// Display messages safely
function showMessage(text) {
    message.replaceChildren();

    const paragraph = document.createElement("p");
    paragraph.textContent = text;

    message.appendChild(paragraph);
    message.style.display = "block";
}

// Convert values into readable text
function displayValue(value) {
    if (Array.isArray(value)) {
        return value
            .map(item => String(item))
            .join(", ");
    }

    if (value === null || value === undefined || value === "") {
        return "Information not available.";
    }

    return String(value);
}

// Convert recipe ingredients into a readable list
function getIngredientList(value) {
    if (Array.isArray(value)) {
        return value
            .map(item => String(item).trim())
            .filter(Boolean);
    }

    const text = displayValue(value);

    if (text === "Information not available.") {
        return [];
    }

    // Some datasets store ingredient arrays as text such as:
    // ['ingredient one', 'ingredient two']
    try {
        const parsed = JSON.parse(text);

        if (Array.isArray(parsed)) {
            return parsed
                .map(item => String(item).trim())
                .filter(Boolean);
        }
    } catch {
        // Continue with normal text handling.
    }

    // Handle a simple Python-style list of quoted strings.
    if (text.startsWith("[") && text.endsWith("]")) {
        const matches = [
            ...text.matchAll(/['"]([^'"]+)['"]/g)
        ];

        if (matches.length > 0) {
            return matches.map(match => match[1].trim());
        }
    }

    return text
        .split(/\s*;\s*/)
        .map(item => item.trim())
        .filter(Boolean);
}

// Convert cooking instructions into numbered steps.
// This is a presentation helper, not an AI rewrite.
function getInstructionSteps(value) {
    const text = displayValue(value);

    if (text === "Information not available.") {
        return [];
    }

    const lines = text
        .split(/\n+/)
        .map(line => line.trim())
        .filter(Boolean);

    // If instructions already contain separate lines, preserve them.
    if (lines.length > 1) {
        return lines.map(line =>
            line.replace(/^(?:step\s*)?\d+[\s.):\-]+/i, "").trim()
        ).filter(Boolean);
    }

    // Otherwise split on sentence endings.
    const sentences = text
        .split(/(?<=[.!?])\s+/)
        .map(sentence => sentence.trim())
        .filter(Boolean);

    return sentences.length > 1 ? sentences : [text];
}

// Create a reusable heading
function createHeading(tag, text) {
    const heading = document.createElement(tag);
    heading.textContent = text;
    return heading;
}

// Create a recipe card with readable sections
function createRecipeCard(recipe, index) {
    const card = document.createElement("article");
    card.className = "recipe-card";

    // Image placeholder
    const emoji = document.createElement("div");
    emoji.className = "recipe-emoji";
    emoji.textContent = "🍲";
    emoji.setAttribute("aria-hidden", "true");

    const content = document.createElement("div");
    content.className = "recipe-content";

    // Recipe title
    const title = createHeading(
        "h3",
        displayValue(recipe.title)
    );

    // Small label
    const label = document.createElement("span");
    label.className = "recipe-label";
    label.textContent = `Recipe ${index + 1}`;

    // Ingredients
    const ingredientsSection = document.createElement("section");
    ingredientsSection.className = "recipe-ingredients";

    ingredientsSection.appendChild(
        createHeading("h4", "🧺 What you'll need")
    );

    const ingredientList = document.createElement("ul");
    ingredientList.className = "ingredient-list";

    const ingredients = getIngredientList(recipe.ingredients);

    if (recipe.matched_ingredients || recipe.missing_ingredients) {
        const matchSummary = document.createElement("p");
        matchSummary.className = "ingredient-match-summary";
        const matched = recipe.matched_ingredients || [];
        const missing = recipe.missing_ingredients || [];
        matchSummary.textContent = matched.length
            ? `You have: ${matched.join(", ")}${missing.length
                ? ` · Still needed: ${missing.join(", ")}`
                : ""}`
            : "None of the requested ingredients were verified in this recipe.";
        ingredientsSection.appendChild(matchSummary);
    }

    if (ingredients.length === 0) {
        const item = document.createElement("li");
        item.textContent = "Ingredients not available.";
        ingredientList.appendChild(item);
    } else {
        ingredients.forEach(ingredient => {
            const item = document.createElement("li");
            item.textContent = ingredient;
            ingredientList.appendChild(item);
        });
    }

    ingredientsSection.appendChild(ingredientList);

    // Cooking instructions
    const details = document.createElement("details");
    details.className = "recipe-instructions";

    const summary = document.createElement("summary");
    summary.textContent = "👩‍🍳 Show easy cooking steps";

    const instructionContent = document.createElement("div");
    instructionContent.className = "instruction-content";

    const steps = getInstructionSteps(recipe.instructions);

    if (steps.length === 0) {
        const paragraph = document.createElement("p");
        paragraph.textContent = "Cooking instructions are not available.";
        instructionContent.appendChild(paragraph);
    } else {
        const orderedList = document.createElement("ol");

        steps.forEach(step => {
            const item = document.createElement("li");
            item.textContent = step;
            orderedList.appendChild(item);
        });

        instructionContent.appendChild(orderedList);
    }

    details.append(summary, instructionContent);

    content.append(
        label,
        title,
        ingredientsSection,
        details
    );

    card.append(emoji, content);

    return card;
}

// Render common Markdown safely as HTML elements.
// All recipe/AI text is inserted through textContent.
function formatAISuggestion(text) {
    const container = document.createElement("div");
    container.className = "ai-formatted-content";

    const lines = String(text)
        .replace(/\r\n/g, "\n")
        .replace(/\*\*|__/g, "")
        .replace(/[–—]/g, "-")
        .split("\n")
        .filter(line => !/^\s*[-*_]{3,}\s*$/.test(line))
        .flatMap(line => {
            if (!line.includes("|")) {
                return [line];
            }

            const cells = line
                .split("|")
                .map(cell => cell.trim());

            while (cells[0] === "") {
                cells.shift();
            }
            while (cells[cells.length - 1] === "") {
                cells.pop();
            }

            const separatorIndex = cells.findIndex(cell =>
                /^:?-{3,}:?$/.test(cell)
            );

            if (separatorIndex <= 0) {
                return [line];
            }

            const columnCount = separatorIndex;
            const expandedLines = [
                `| ${cells.slice(0, columnCount).join(" | ")} |`,
                `| ${cells
                    .slice(separatorIndex, separatorIndex + columnCount)
                    .join(" | ")} |`
            ];

            const dataCells = cells.slice(separatorIndex + columnCount);
            for (let index = 0; index < dataCells.length; index += columnCount) {
                const row = dataCells.slice(index, index + columnCount);
                if (row.length === columnCount) {
                    expandedLines.push(`| ${row.join(" | ")} |`);
                }
            }

            return expandedLines;
        });

    let list = null;
    let paragraphLines = [];
    let activeTable = null;

    function appendInlineText(parent, value) {
        const pattern = /(\*\*|__)(.+?)\1|(\*|_)(.+?)\3/g;
        let lastIndex = 0;
        let match;

        while ((match = pattern.exec(value)) !== null) {
            if (match.index > lastIndex) {
                parent.appendChild(
                    document.createTextNode(value.slice(lastIndex, match.index))
                );
            }

            const strong = match[1] !== undefined;
            const element = document.createElement(strong ? "strong" : "em");
            element.textContent = strong ? match[2] : match[4];
            parent.appendChild(element);
            lastIndex = pattern.lastIndex;
        }

        if (lastIndex < value.length) {
            parent.appendChild(
                document.createTextNode(value.slice(lastIndex))
            );
        }
    }

    function appendInline(parent, value) {
        const codeParts = String(value).split("`");
        codeParts.forEach((part, index) => {
            if (index % 2 === 1) {
                const code = document.createElement("code");
                code.textContent = part;
                parent.appendChild(code);
            } else {
                appendInlineText(parent, part);
            }
        });
    }

    function flushParagraph() {
        if (paragraphLines.length === 0) return;

        const paragraph = document.createElement("p");
        appendInline(paragraph, paragraphLines.join(" "));
        container.appendChild(paragraph);
        paragraphLines = [];
    }

    function closeList() {
        list = null;
    }

    function closeTable() {
        activeTable = null;
    }

    lines.forEach(rawLine => {
        const line = rawLine.trim();

        if (!line) {
            flushParagraph();
            closeList();
            closeTable();
            return;
        }

        // Headings
        const headingMatch = line.match(/^(#{1,3})\s+(.+)$/);

        if (headingMatch) {
            flushParagraph();
            closeList();
            closeTable();

            const heading = document.createElement(
                headingMatch[1].length === 1 ? "h2" : "h3"
            );

            appendInline(heading, headingMatch[2]);
            container.appendChild(heading);
            return;
        }

        // Bullet lists
        const bulletMatch = line.match(/^[-*•]\s+(.+)$/);

        // Numbered lists
        const numberMatch = line.match(/^\d+[.)]\s+(.+)$/);

        if (bulletMatch || numberMatch) {
            flushParagraph();
            closeTable();

            const listType = numberMatch ? "ol" : "ul";

            if (!list || list.tagName.toLowerCase() !== listType) {
                list = document.createElement(listType);
                container.appendChild(list);
            }

            const item = document.createElement("li");
            appendInline(item, (bulletMatch || numberMatch)[1]);
            list.appendChild(item);
            return;
        }

        // Markdown tables are rendered as DOM nodes, never as HTML strings.
        if (line.includes("|")) {
            const cells = line
                .split("|")
                .map(cell => cell.trim())
                .filter((cell, index, allCells) =>
                    !(index === 0 && cell === "") &&
                    !(index === allCells.length - 1 && cell === "")
                );
            if (cells.length > 1 && cells.every(Boolean)) {
                if (cells.every(cell => /^:?-{3,}:?$/.test(cell))) {
                    return;
                }
                flushParagraph();
                closeList();

                if (!activeTable) {
                    activeTable = document.createElement("table");
                    const body = document.createElement("tbody");
                    activeTable.appendChild(body);
                    container.appendChild(activeTable);
                }

                const body = activeTable.querySelector("tbody");
                const row = document.createElement("tr");
                cells.forEach(cell => {
                    const tableCell = body.children.length === 0
                        ? document.createElement("th")
                        : document.createElement("td");
                    appendInline(tableCell, cell);
                    row.appendChild(tableCell);
                });
                body.appendChild(row);
                return;
            }
        }

        closeList();
        closeTable();
        paragraphLines.push(line);
    });

    flushParagraph();

    return container;
}

// Create the AI suggestion section
function createAISuggestionSection() {
    let section = document.getElementById("aiSuggestionSection");

    if (!section) {
        section = document.createElement("section");
        section.id = "aiSuggestionSection";
        section.className = "ai-suggestion-section";

        const topLabel = document.createElement("span");
        topLabel.className = "ai-label";
        topLabel.textContent = "POWERED BY GROQ AI";

        const heading = createHeading(
            "h2",
            "✨ Your personalized recipe idea"
        );

        const description = document.createElement("p");
        description.className = "ai-description";
        description.textContent =
            "A simple suggestion based on your search and the recipes found.";

        const output = document.createElement("div");
        output.id = "aiSuggestionOutput";
        output.setAttribute("aria-live", "polite");

        section.append(topLabel, heading, description, output);

        // Place AI suggestion above recipe cards.
        recipeResults.parentNode.insertBefore(section, recipeResults);
    }

    section.style.display = "block";

    return document.getElementById("aiSuggestionOutput");
}

// Request an AI suggestion from Flask/Groq
async function getAISuggestion(query, recipes, userIngredients, searchId) {
    const output = createAISuggestionSection();

    output.replaceChildren();

    const loading = document.createElement("p");
    loading.className = "ai-loading";
    loading.textContent = "🤖 Thinking of a delicious idea...";
    output.appendChild(loading);

    try {
        const response = await fetch(`${API_URL}/api/ai-suggest`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                query,
                ingredients: userIngredients,
                recipes: recipes.slice(0, 5)
            })
        });

        let data;
        try {
            data = await response.json();
        } catch {
            throw new Error("The AI returned an invalid response.");
        }

        if (!response.ok || data.success === false) {
            throw new Error(
                data.error || "The AI could not generate a suggestion."
            );
        }

        // Ignore an older response if the user has searched again.
        if (searchId !== currentSearchId) return;

        output.replaceChildren();

        const suggestion = data.suggestion;

        if (!suggestion) {
            const paragraph = document.createElement("p");
            paragraph.textContent = "No AI suggestion was returned.";
            output.appendChild(paragraph);
            return;
        }

        output.appendChild(formatAISuggestion(suggestion));

        const note = document.createElement("p");
        note.className = "ai-note";
        note.textContent =
            "Tip: Check the ingredients and cooking instructions before preparing the dish.";

        output.appendChild(note);

    } catch (error) {
        console.error("AI suggestion error:", error);

        if (searchId !== currentSearchId) return;

        output.replaceChildren();

        const paragraph = document.createElement("p");
        paragraph.className = "ai-error";
        paragraph.textContent =
            "The recipes were found, but the AI suggestion is unavailable right now. Please try again in a moment.";

        output.appendChild(paragraph);
    }
}

// Search recipes using FAISS
async function searchRecipes(query) {
    query = query.trim();

    if (!query) {
        showMessage("Please enter a dish or ingredient to search.");
        searchInput.focus();
        return;
    }

    const searchId = ++currentSearchId;

    searchButton.disabled = true;
    searchButton.textContent = "Searching...";

    recipeResults.replaceChildren();
    resultCount.textContent = "";

    const previousAISection =
        document.getElementById("aiSuggestionSection");

    if (previousAISection) {
        previousAISection.style.display = "none";
    }

    showMessage("Finding recipes for you...");

    resultsTitle.textContent = "Recipes picked for you";
    resultsInfo.textContent = `Results related to "${query}"`;

    try {
        const url = new URL(`${API_URL}/api/search`);
        url.searchParams.set("q", query);
        url.searchParams.set("limit", "9");

        const response = await fetch(url);
        let data;
        try {
            data = await response.json();
        } catch {
            throw new Error("The recipe search returned an invalid response.");
        }

        // Ignore stale search responses.
        if (searchId !== currentSearchId) return;

        if (!response.ok) {
            throw new Error(data.error || "Recipe search failed.");
        }

        if (!data.results || data.results.length === 0) {
            showMessage(
                "We couldn't find recipes for that search. Try an ingredient like chicken, potatoes, or pasta."
            );
            return;
        }

        message.style.display = "none";

        resultCount.textContent =
            `${data.count ?? data.results.length} recipes found`;

        data.results.forEach((recipe, index) => {
            recipeResults.appendChild(
                createRecipeCard(recipe, index)
            );
        });

        // Load AI suggestion while keeping recipe cards visible.
        getAISuggestion(
            query,
            data.results,
            data.user_ingredients || [],
            searchId
        );

    } catch (error) {
        console.error("Recipe search error:", error);

        if (searchId === currentSearchId) {
            showMessage(
                "Couldn't connect to RecipeFinder AI. Please check that the backend is running and try again."
            );
        }
    } finally {
        // Only update the button for the latest search.
        if (searchId === currentSearchId) {
            searchButton.disabled = false;
            searchButton.textContent = "Find Recipes →";
        }
    }
}

// Search form submission
searchForm.addEventListener("submit", event => {
    event.preventDefault();
    searchRecipes(searchInput.value);
});

// Popular search buttons
document.querySelectorAll(".suggestions button").forEach(button => {
    button.addEventListener("click", () => {
        const query = button.dataset.query;

        if (!query) return;

        searchInput.value = query;
        searchRecipes(query);
    });
});
