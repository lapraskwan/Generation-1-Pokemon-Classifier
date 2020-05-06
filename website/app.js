var express = require('express');
var app = express();

// Serve static files
app.use('/tfjs_model', express.static(__dirname + '/tfjs_model'));
app.use('/label_name.json', express.static(__dirname + '/label_name.json'));

// Routes
app.get('/', function (req, res) {
    res.sendFile(__dirname + '/index.html');
});

app.listen(process.env.PORT || 3000, function () {
    console.log('Example app listening on port 3000!');
});