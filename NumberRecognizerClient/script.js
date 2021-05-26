var canvas, ctx, flag = false,
        prevX = 0,
        currX = 0,
        prevY = 0,
        currY = 0,
        dot_flag = false;

var x = "black",
    y = 20;


var transforming = false;
var serverUrl = "http://localhost:3030";

function draw() {
    ctx.beginPath();
    ctx.moveTo(prevX, prevY);
    ctx.lineTo(currX, currY);
    ctx.strokeStyle = x;
    ctx.lineWidth = y;
    ctx.lineCap = 'round';
    ctx.stroke();
    ctx.closePath();
}

function erase() {
    ctx.clearRect(0, 0, w, h);
    document.getElementById("respDiv").classList.add("hidden");
}

function findxy(res, e) {
    if (res == 'down') {
        prevX = currX;
        prevY = currY;
        currX = e.clientX - canvas.offsetLeft;
        currY = e.clientY - canvas.offsetTop;

        flag = true;
        dot_flag = true;
        if (dot_flag) {
            ctx.beginPath();
            ctx.fillStyle = x;
            ctx.fillRect(currX, currY, 2, 2);
            ctx.closePath();
            dot_flag = false;
        }
    }
    if (res == 'up' || res == "out") {
        flag = false;
        var data = getPredictionData();
        getPrediction(data);
    }
    if (res == 'move') {
        if (flag) {
            prevX = currX;
            prevY = currY;
            currX = e.clientX - canvas.offsetLeft;
            currY = e.clientY - canvas.offsetTop;
            draw();
        }
    }
}


function getPredictionData() {
    if (transforming)
        return;
    else
        transforming = true;

    var width = 28;
    var height = 28;
    var multi = 8;
    var imgData = ctx.getImageData(0, 0, width * multi, height * multi).data;
    var data = [];

    // remove rgb and keep alpha
    for (var i = 3, n = imgData.length; i < n; i += 4) {
        data.push(imgData[i]);
    }

    // transform into 28x28
    var dataTransformed = [];
    var m = Math.pow(multi, 2);
    for (var row = 0; row < data.length ; row += width * m) {
        for (var col = row; col < row + width * multi ; col += multi) {
            var pixelSum = 0;
            for (var i = 0; i < multi; i++) {
                for (var j = 0; j < multi; j++) {
                    var s = col + i + width * multi * j;
                    pixelSum += data[s];
                }
            }
            dataTransformed.push(Math.floor(pixelSum / m));
        }
    }

    transforming = false;
    return dataTransformed;
}


function getPrediction(pixelData) {
    var xhr = new XMLHttpRequest();
    xhr.open("POST", serverUrl, true);
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.onreadystatechange = function () {
        if (xhr.readyState === 4 && xhr.status === 200) {
            var json = JSON.parse(xhr.responseText);
            document.getElementById("number").textContent = json.prediction;
            document.getElementById("acc").textContent = json.accuracy[json.prediction];
            console.log(json);
            var acc = json.accuracy;
            var moreAcc = document.getElementById("moreAcc");
            moreAcc.innerHTML = '';
            for (var i = 0; i < 10; i++) {
                var p = document.createElement("p");
                p.textContent = `${i}: ${acc[i]}%`;
                moreAcc.appendChild(p);
            }
            document.getElementById("respDiv").classList.remove("hidden");
        }
    };
    var data = JSON.stringify({"pixels": pixelData});
    xhr.send(data);
}



window.onload = function() {
    canvas = document.getElementById('can');
    ctx = canvas.getContext("2d");
    w = canvas.width;
    h = canvas.height;

    canvas.addEventListener("mousemove", function (e) {
        findxy('move', e)
    }, false);
    canvas.addEventListener("mousedown", function (e) {
        findxy('down', e)
    }, false);
    canvas.addEventListener("mouseup", function (e) {
        findxy('up', e)
    }, false);
    canvas.addEventListener("mouseout", function (e) {
        findxy('out', e)
    }, false);
}
