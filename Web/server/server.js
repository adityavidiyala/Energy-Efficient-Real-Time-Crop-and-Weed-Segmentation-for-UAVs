const express = require('express');
const cors = require('cors');
const { selectConfig } = require('./logic');

const app = express();
app.use(cors());
app.use(express.json());

// API Endpoint
app.post('/api/rdd-status', (req, res) => {
    const { battery, velocity, temp, latencyManual } = req.body;

    console.log("Received inputs:", req.body); // Debugging log

    const result = selectConfig(
        parseFloat(latencyManual), 
        parseFloat(battery), 
        parseFloat(velocity), 
        parseFloat(temp)
    );

    res.json(result);
});

const PORT = 5000;
app.listen(PORT, () => console.log(`🧠 Node Brain running on port ${PORT}`));