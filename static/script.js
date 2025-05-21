const editor = document.getElementById("editor");
let suggestion = "";
let isSuggesting = false;

function getEditorText() {
    // Get only the user-typed text (ignore suggestion spans)
    const textNodes = [...editor.childNodes].filter(n => n.nodeType === Node.TEXT_NODE);
    return textNodes.map(n => n.textContent).join('').trim();
}

function showSuggestion(text) {
    // Clear existing content
    editor.innerHTML = '';
    
    // Add user text with space
    const textNode = document.createTextNode(text + (text.length ? " " : ""));
    editor.appendChild(textNode);
    
    // Add suggestion if available
    if (suggestion) {
        const suggestionSpan = document.createElement('span');
        suggestionSpan.className = 'suggestion';
        suggestionSpan.textContent = suggestion;
        editor.appendChild(suggestionSpan);
    }
    
    // Place cursor after user text (before suggestion)
    placeCursorAfterText(text);
}

function placeCursorAfterText(text) {
    const range = document.createRange();
    const sel = window.getSelection();
    const textNode = [...editor.childNodes].find(n => n.nodeType === Node.TEXT_NODE);
    const cursorPos = text.length + (text.length ? 1 : 0); // +1 for space if text exists
    range.setStart(textNode || editor, Math.min(cursorPos, (textNode?.length || 0)));
    range.collapse(true);
    sel.removeAllRanges();
    sel.addRange(range);
}

async function fetchSuggestion(text) {
    try {
        const response = await fetch(`/suggest?text=${encodeURIComponent(text)}`);
        return response.ok ? await response.text() : "";
    } catch (error) {
        console.error("Error:", error);
        return "";
    }
}

editor.addEventListener("input", () => {
    // Clear suggestion if user types over it
    if (isSuggesting && !editor.innerHTML.includes('suggestion')) {
        isSuggesting = false;
        suggestion = "";
    }
});

editor.addEventListener("keydown", async (e) => {
    const currentText = getEditorText();
    
    // TAB: Accept suggestion or get new one
    if (e.key === "Tab") {
        e.preventDefault();
    
        const current = getEditorText();
    
        if (isSuggesting) {
            // Accept current suggestion
            const confirmedText = current + " " + suggestion;
            editor.textContent = confirmedText + " ";
            isSuggesting = false;
            suggestion = "";
            placeCursorAfterText(confirmedText);
        } else {
            suggestion = await fetchSuggestion(current);
            if (suggestion) {
                isSuggesting = true;
                // Immediately accept suggestion
                const confirmedText = current + " " + suggestion;
                editor.textContent = confirmedText + " ";
                isSuggesting = false;
                suggestion = "";
                placeCursorAfterText(confirmedText);
            }
        }
        return;
    }
    
    
    // SPACE: Insert space and show suggestion
    if (e.key === " ") {
        e.preventDefault();
        const newText = currentText + " ";
        editor.textContent = newText;
        suggestion = await fetchSuggestion(newText.trim());
        if (suggestion) {
            isSuggesting = true;
            showSuggestion(newText);
        }
        return;
    }
});

// Handle regular typing
editor.addEventListener("keypress", (e) => {
    if (!['Tab', ' '].includes(e.key)) {
        isSuggesting = false;
        suggestion = "";
        // Ensure cursor stays after typed text
        setTimeout(() => placeCursorAfterText(getEditorText()), 0);
    }
});

window.onload = () => {
    editor.focus();
    // Initialize with empty content and proper cursor
    editor.textContent = "";
    placeCursorAfterText("");
};