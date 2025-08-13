let selectedFile = null;

// File input handling
document.getElementById('fileInput').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = function(e) {
            const preview = document.getElementById('previewImage');
            preview.src = e.target.result;
            preview.style.display = 'block';
        };
        reader.readAsDataURL(file);
        document.getElementById('analyzeBtn').disabled = false;
    }
});

// Drag and drop
const uploadSection = document.querySelector('.upload-section');

uploadSection.addEventListener('dragover', function(e) {
    e.preventDefault();
    uploadSection.classList.add('dragover');
});

uploadSection.addEventListener('dragleave', function(e) {
    e.preventDefault();
    uploadSection.classList.remove('dragover');
});

uploadSection.addEventListener('drop', function(e) {
    e.preventDefault();
    uploadSection.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        const file = files[0];
        if (file.type.startsWith('image/')) {
            selectedFile = file;
            const reader = new FileReader();
            reader.onload = function(e) {
                const preview = document.getElementById('previewImage');
                preview.src = e.target.result;
                preview.style.display = 'block';
            };
            reader.readAsDataURL(file);
            document.getElementById('analyzeBtn').disabled = false;
        }
    }
});

// Analyze button
document.getElementById('analyzeBtn').addEventListener('click', async function() {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('occasion', document.getElementById('occasion').value);
    formData.append('include_suggestions', 'true');
    
    const stylePreference = document.getElementById('style_preference').value;
    if (stylePreference) {
        formData.append('user_style_preference', stylePreference);
    }

    // Show loading
    document.getElementById('loading').style.display = 'block';
    document.getElementById('results').style.display = 'none';
    document.getElementById('errorMessage').style.display = 'none';
    document.getElementById('analyzeBtn').disabled = true;

    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        displayResults(result);

    } catch (error) {
        console.error('Error:', error);
        document.getElementById('errorMessage').textContent = 'Error analyzing outfit: ' + error.message;
        document.getElementById('errorMessage').style.display = 'block';
    } finally {
        document.getElementById('loading').style.display = 'none';
        document.getElementById('analyzeBtn').disabled = false;
    }
});

function displayResults(result) {
    // Display score
    document.getElementById('scoreNumber').textContent = result.style_score;
    document.getElementById('scoreText').textContent = result.contextual_feedback;

    // Display detected items
    displayDetectedItems(result.detected_items);

    // Display breakdown
    const breakdown = document.getElementById('breakdown');
    breakdown.innerHTML = '';
    
    const breakdownData = [
        { name: 'Contextual', score: result.scoring_breakdown.clip_contextual },
        { name: 'Color Harmony', score: result.scoring_breakdown.color_harmony },
        { name: 'Completeness', score: result.scoring_breakdown.item_completeness },
        { name: 'Coherence', score: result.scoring_breakdown.style_coherence }
    ];

    breakdownData.forEach(item => {
        const div = document.createElement('div');
        div.className = 'breakdown-item';
        div.innerHTML = `
            <div class="breakdown-score">${item.score}</div>
            <div>${item.name}</div>
        `;
        breakdown.appendChild(div);
    });

    // Display suggestions
    const suggestionsList = document.getElementById('suggestionsList');
    suggestionsList.innerHTML = '';

    if (result.whats_working) {
        const div = document.createElement('div');
        div.className = 'suggestion-item';
        div.innerHTML = `<strong>✅ What's Working:</strong> ${result.whats_working}`;
        suggestionsList.appendChild(div);
    }

    if (result.areas_for_improvement) {
        const div = document.createElement('div');
        div.className = 'suggestion-item';
        div.innerHTML = `<strong>🔧 Areas for Improvement:</strong> ${result.areas_for_improvement}`;
        suggestionsList.appendChild(div);
    }

    if (result.specific_suggestions && result.specific_suggestions.length > 0) {
        result.specific_suggestions.forEach(suggestion => {
            const div = document.createElement('div');
            div.className = 'suggestion-item';
            div.innerHTML = `<strong>💡 Suggestion:</strong> ${suggestion}`;
            suggestionsList.appendChild(div);
        });
    }

    if (result.occasion_tips) {
        const div = document.createElement('div');
        div.className = 'suggestion-item';
        div.innerHTML = `<strong>🎯 Occasion Tips:</strong> ${result.occasion_tips}`;
        suggestionsList.appendChild(div);
    }

    document.getElementById('results').style.display = 'block';
}

function displayDetectedItems(detectedItems) {
    const itemsGrid = document.getElementById('itemsGrid');
    itemsGrid.innerHTML = '';

    if (!detectedItems || detectedItems.length === 0) {
        itemsGrid.innerHTML = '<p>No clothing items detected</p>';
        return;
    }

    detectedItems.forEach(item => {
        const itemCard = document.createElement('div');
        itemCard.className = 'item-card';

        // Create color tags
        let colorTagsHTML = '';
        if (item.colors && item.colors.length > 0) {
            colorTagsHTML = item.colors.map(color => 
                `<span class="color-tag ${color.name.toLowerCase().replace('_', '-')}">${color.name.replace('_', ' ')}</span>`
            ).join('');
        } else {
            colorTagsHTML = '<span class="color-tag">No colors detected</span>';
        }

        // Get item emoji
        const itemEmojis = {
            'shirt': '👔',
            'pants': '👖', 
            'jacket': '🧥',
            'dress': '👗',
            'skirt': '🩱',
            'shorts': '🩳',
            'shoe': '👟',
            'bag': '👜',
            'hat': '👒',
            'sunglass': '🕶️'
        };

        const emoji = itemEmojis[item.class] || '👕';

        itemCard.innerHTML = `
            <div class="item-name">${emoji} ${item.class}</div>
            <div class="item-confidence">Confidence: ${Math.round(item.confidence * 100)}%</div>
            <div class="colors-section">
                <div class="colors-label">Detected Colors:</div>
                <div class="color-tags">
                    ${colorTagsHTML}
                </div>
            </div>
        `;

        itemsGrid.appendChild(itemCard);
    });
}