const tf = require('@tensorflow/tfjs-node');
const { load } = require('@tensorflow/tfjs-node');
const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const { WordNetLemmatizer } = require('nltk');

const app = express();
const PORT = process.env.PORT || 3000;

// Load the model and intents file
let model;
let intents;

async function loadModel() {
    // Load TensorFlow model
    model = await load('model/intent_model.h5');
    // Load intents data
    intents = JSON.parse(fs.readFileSync('model/correctedIntents.json', 'utf-8'));
}

loadModel();

app.use(cors());
app.use(bodyParser.json());

// Helper functions
const lemmatizer = new WordNetLemmatizer();

function preprocessInput(text) {
    const words = text.split(/\s+/);
    return words.map(word => lemmatizer.lemmatize(word.toLowerCase()));
}

function bowOfWords(words_input, words) {
    let bag = new Array(words.length).fill(0);
    words_input.forEach(w => {
        const index = words.indexOf(w);
        if (index >= 0) bag[index] = 1;
    });
    return bag;
}

app.post('/api/chat', async (req, res) => {
    const userInput = req.body.text;
    const words = []; // Array to store all words used in training
    const classes = []; // Array to store all possible intent tags
    
    // Populate words and classes from intents file
    intents.intents.forEach(intent => {
        intent.patterns.forEach(pattern => {
            const wordsInput = preprocessInput(pattern);
            words.push(...wordsInput);
        });
        if (!classes.includes(intent.tag)) {
            classes.push(intent.tag);
        }
    });
    
    const processedInput = preprocessInput(userInput);
    const bowInput = bowOfWords(processedInput, words);
    
    // Predict the class
    const prediction = model.predict(tf.tensor([bowInput]));
    const intentIndex = prediction.argMax(-1).dataSync()[0];
    const intentTag = classes[intentIndex];
    
    // Get the response from intents data
    const intent = intents.intents.find(i => i.tag === intentTag);
    const response = intent ? intent.responses[Math.floor(Math.random() * intent.responses.length)] : "Sorry, I didn't get that.";
    
    res.json({ response });
});

app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});
