const API_BASE = "/api";
let currentTicker = "";
let chartInstance = null;

// Helper to format currency numbers in Vietnamese format
function formatCurrency(value, currency = "VND") {
    if (value === null || value === undefined) return "---";
    if (currency.toUpperCase() === "USD") {
        return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" }).format(value);
    }
    return new Intl.NumberFormat("vi-VN", { style: "currency", currency: "VND" }).format(value).replace("₫", "VNĐ");
}

// Helper to check risk level classes
function getRiskClass(level) {
    if (level.includes("Thấp") || level.includes("Low")) return "risk-low";
    if (level.includes("Trung bình") || level.includes("Medium")) return "risk-medium";
    return "risk-high";
}

// App Initialization
document.addEventListener("DOMContentLoaded", async () => {
    await loadWatchlist();
    
    // Bind predict button
    document.getElementById("btn-trigger-predict").addEventListener("click", triggerPredict);
});

// Load and populate Watchlist
async function loadWatchlist() {
    const listContainer = document.getElementById("ticker-list");
    try {
        const response = await fetch(`${API_BASE}/stocks`);
        if (!response.ok) throw new Error("Không thể tải danh sách mã");
        
        const stocks = await response.json();
        listContainer.innerHTML = "";
        
        if (stocks.length === 0) {
            listContainer.innerHTML = `<div class="ticker-item text-center">Trống. Vui lòng import dữ liệu.</div>`;
            return;
        }
        
        for (const stock of stocks) {
            const item = document.createElement("div");
            item.className = "ticker-item";
            item.dataset.ticker = stock.ticker;
            item.dataset.currency = stock.currency;
            
            // Get latest close price for the watchlist list item
            let priceText = "Đang tải...";
            let changeText = "";
            let changeClass = "";
            
            try {
                const priceRes = await fetch(`${API_BASE}/prices/${stock.ticker}?limit=2`);
                if (priceRes.ok) {
                    const priceData = await priceRes.json();
                    if (priceData.length > 0) {
                        const latest = priceData[priceData.length - 1];
                        priceText = formatCurrency(latest.close, stock.currency);
                        
                        if (priceData.length > 1) {
                            const prev = priceData[priceData.length - 2];
                            const pct = ((latest.close - prev.close) / prev.close) * 100;
                            changeClass = pct >= 0 ? "up" : "down";
                            changeText = `${pct >= 0 ? "+" : ""}${pct.toFixed(2)}%`;
                        }
                    } else {
                        priceText = "No data";
                    }
                }
            } catch (err) {
                console.error("Error fetching item price:", err);
            }
            
            item.innerHTML = `
                <div class="ticker-info">
                    <span class="ticker-symbol">${stock.ticker}</span>
                    <span class="ticker-name">${stock.name}</span>
                </div>
                <div class="ticker-stats">
                    <span class="ticker-price">${priceText}</span>
                    ${changeText ? `<span class="ticker-change ${changeClass}">${changeText}</span>` : ""}
                </div>
            `;
            
            item.addEventListener("click", () => selectTicker(stock.ticker));
            listContainer.appendChild(item);
        }
        
        // Select first stock by default
        if (stocks.length > 0) {
            selectTicker(stocks[0].ticker);
        }
        
        // Highlight active stock symbol again if currentTicker is set
        if (currentTicker) {
            document.querySelectorAll(".ticker-item").forEach(item => {
                if (item.dataset.ticker === currentTicker) {
                    item.classList.add("active");
                } else {
                    item.classList.remove("active");
                }
            });
        }
        
    } catch (error) {
        listContainer.innerHTML = `<div class="ticker-item text-center text-red">Lỗi: ${error.message}</div>`;
    }
}

// Select stock and reload dashboard data
async function selectTicker(ticker) {
    currentTicker = ticker;
    
    // Highlight sidebar active item
    document.querySelectorAll(".ticker-item").forEach(item => {
        if (item.dataset.ticker === ticker) {
            item.classList.add("active");
        } else {
            item.classList.remove("active");
        }
    });
    
    await refreshDashboard();
}

// Trigger refresh
async function refreshDashboard() {
    if (!currentTicker) return;
    
    const tickerItem = document.querySelector(`.ticker-item[data-ticker="${currentTicker}"]`);
    const currency = tickerItem ? tickerItem.dataset.currency : "VND";
    
    try {
        // Fetch APIs
        const [prices, predictions, news] = await Promise.all([
            fetch(`${API_BASE}/prices/${currentTicker}?limit=150`).then(res => res.json()),
            fetch(`${API_BASE}/predictions/${currentTicker}?limit=10`).then(res => res.json()),
            fetch(`${API_BASE}/news/${currentTicker}?limit=10`).then(res => res.json())
        ]);
        
        // Update USD rate container
        const usdRateElement = document.getElementById("usd-vnd-rate");
        if (predictions.length > 0 && predictions[0].usd_vnd_rate) {
            usdRateElement.textContent = formatCurrency(predictions[0].usd_vnd_rate, "VND");
        }
        
        // Update header block
        const titleEl = document.getElementById("selected-stock-title");
        const exchangeEl = document.getElementById("selected-stock-exchange");
        const priceEl = document.getElementById("selected-stock-price");
        const changeEl = document.getElementById("selected-stock-change");
        const riskEl = document.getElementById("selected-stock-risk");
        const dateEl = document.getElementById("selected-stock-date");
        
        titleEl.textContent = currentTicker;
        exchangeEl.textContent = currentTicker.includes(".VN") ? "Sở giao dịch Chứng khoán TP.HCM (HOSE)" : "Sở giao dịch Chứng khoán New York (NYSE / NASDAQ)";
        
        if (prices.length > 0) {
            const latest = prices[prices.length - 1];
            priceEl.textContent = formatCurrency(latest.close, currency);
            dateEl.textContent = latest.date;
            
            if (prices.length > 1) {
                const prev = prices[prices.length - 2];
                const pct = ((latest.close - prev.close) / prev.close) * 100;
                changeEl.textContent = `${pct >= 0 ? "📈 +" : "📉 "}${pct.toFixed(2)}%`;
                changeEl.className = `stock-change ${pct >= 0 ? "up" : "down"}`;
            }
        } else {
            priceEl.textContent = "---";
            changeEl.textContent = "";
            dateEl.textContent = "---";
        }
        
        // Update predictions card
        const xgbTrend = document.getElementById("xgb-trend");
        const xgbPrice = document.getElementById("xgb-price");
        const xgbPriceUsd = document.getElementById("xgb-price-usd");
        const xgbRange = document.getElementById("xgb-range");
        
        const transTrend = document.getElementById("trans-trend");
        const transPrice = document.getElementById("trans-price");
        const transPriceUsd = document.getElementById("trans-price-usd");
        const transRange = document.getElementById("trans-range");
        
        const predDate = document.getElementById("pred-date");
        const targetDate = document.getElementById("target-date");
        
        if (predictions.length > 0) {
            const latestPred = predictions[0];
            const lastClose = prices.length > 0 ? prices[prices.length - 1].close : latestPred.xgb_predicted_price;
            
            riskEl.textContent = latestPred.risk_level;
            riskEl.className = `badge-val ${getRiskClass(latestPred.risk_level)}`;
            
            // XGBoost
            const xgbDiff = ((latestPred.xgb_predicted_price - lastClose) / lastClose) * 100;
            xgbTrend.textContent = xgbDiff >= 0 ? `📈 TĂNG (${xgbDiff.toFixed(2)}%)` : `📉 GIẢM (${xgbDiff.toFixed(2)}%)`;
            xgbTrend.className = `trend-indicator ${xgbDiff >= 0 ? "up" : "down"}`;
            xgbPrice.textContent = formatCurrency(latestPred.xgb_predicted_price, currency);
            xgbRange.textContent = `[${formatCurrency(latestPred.xgb_lower, currency)} - ${formatCurrency(latestPred.xgb_upper, currency)}]`;
            
            // Transformer
            const transDiff = ((latestPred.trans_predicted_price - lastClose) / lastClose) * 100;
            transTrend.textContent = transDiff >= 0 ? `📈 TĂNG (${transDiff.toFixed(2)}%)` : `📉 GIẢM (${transDiff.toFixed(2)}%)`;
            transTrend.className = `trend-indicator ${transDiff >= 0 ? "up" : "down"}`;
            transPrice.textContent = formatCurrency(latestPred.trans_predicted_price, currency);
            transRange.textContent = `[${formatCurrency(latestPred.trans_lower, currency)} - ${formatCurrency(latestPred.trans_upper, currency)}]`;
            
            // Meta USD Conversion if currency is VND and not VNM.VN
            if (currency.toUpperCase() === "VND" && latestPred.usd_vnd_rate && !currentTicker.includes("VNM")) {
                const rate = latestPred.usd_vnd_rate;
                xgbPriceUsd.style.display = "block";
                xgbPriceUsd.textContent = `$${(latestPred.xgb_predicted_price / rate).toFixed(2)} USD`;
                transPriceUsd.style.display = "block";
                transPriceUsd.textContent = `$${(latestPred.trans_predicted_price / rate).toFixed(2)} USD`;
            } else {
                xgbPriceUsd.style.display = "none";
                transPriceUsd.style.display = "none";
            }
            
            predDate.textContent = latestPred.prediction_date;
            targetDate.textContent = latestPred.target_date;
        } else {
            riskEl.textContent = "---";
            riskEl.className = "badge-val";
            
            xgbTrend.textContent = "--";
            xgbPrice.textContent = "---";
            xgbRange.textContent = "[---]";
            xgbPriceUsd.style.display = "none";
            
            transTrend.textContent = "--";
            transPrice.textContent = "---";
            transRange.textContent = "[---]";
            transPriceUsd.style.display = "none";
            
            predDate.textContent = "---";
            targetDate.textContent = "---";
        }
        
        // Update news feed
        const newsContainer = document.getElementById("news-list");
        newsContainer.innerHTML = "";
        
        if (news.length === 0) {
            newsContainer.innerHTML = `<div class="news-empty">Không có phân tích cảm xúc tin tức nào gần đây.</div>`;
        } else {
            news.forEach(item => {
                const nItem = document.createElement("div");
                nItem.className = "news-item";
                
                const labelClass = item.sentiment_label === "positive" ? "positive" : (item.sentiment_label === "negative" ? "negative" : "neutral");
                const labelText = item.sentiment_label === "positive" ? "Tích cực" : (item.sentiment_label === "negative" ? "Tiêu cực" : "Trung tính");
                
                nItem.innerHTML = `
                    <div class="news-header">
                        <span>${item.source || "Nguồn tin"} | ${item.published_date}</span>
                    </div>
                    <div class="news-title">${item.title}</div>
                    <div class="news-footer">
                        <span class="news-sentiment ${labelClass}">${labelText}</span>
                        <span class="news-score">Điểm số: ${item.sentiment_score.toFixed(4)}</span>
                    </div>
                `;
                newsContainer.appendChild(nItem);
            });
        }
        
        // Render Chart.js
        renderChart(prices, predictions[0], currency);
        
    } catch (err) {
        console.error("Lỗi khi load dữ liệu chi tiết:", err);
    }
}

// Render line chart with forecast endpoints and margins
// Render line chart with forecast endpoints and margins
function renderChart(prices, latestPrediction, currency) {
    const ctx = document.getElementById("priceChart").getContext("2d");
    
    if (chartInstance) {
        chartInstance.destroy();
    }
    
    // Last 30 sessions of prices
    const sliceCount = 30;
    const recentPrices = prices.slice(-sliceCount);
    
    const labels = recentPrices.map(p => p.date);
    const rawData = recentPrices.map(p => p.close);
    
    const xgbData = Array(recentPrices.length).fill(null);
    const transData = Array(recentPrices.length).fill(null);
    
    // Append prediction if available
    if (latestPrediction && recentPrices.length > 0) {
        const lastIdx = recentPrices.length - 1;
        const lastPrice = recentPrices[lastIdx].close;
        
        labels.push(latestPrediction.target_date);
        rawData.push(null); // No actual close price for target date yet
        
        // Connecting lines start from last close
        xgbData[lastIdx] = lastPrice;
        xgbData.push(latestPrediction.xgb_predicted_price);
        
        transData[lastIdx] = lastPrice;
        transData.push(latestPrediction.trans_predicted_price);
    }
    
    chartInstance = new Chart(ctx, {
        type: "line",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "Giá đóng cửa thực tế",
                    data: rawData,
                    borderColor: "#3b82f6",
                    backgroundColor: "rgba(59, 130, 246, 0.05)",
                    borderWidth: 3,
                    pointBackgroundColor: "#3b82f6",
                    pointRadius: 3,
                    fill: true,
                    tension: 0.1
                },
                {
                    label: "Dự đoán XGBoost",
                    data: xgbData,
                    borderColor: "#2ea043",
                    borderWidth: 2.5,
                    borderDash: [5, 5],
                    pointBackgroundColor: "#2ea043",
                    pointRadius: 5,
                    pointStyle: "circle",
                    tension: 0,
                    fill: false
                },
                {
                    label: "Dự đoán Transformer",
                    data: transData,
                    borderColor: "#a371f7",
                    borderWidth: 2.5,
                    borderDash: [5, 5],
                    pointBackgroundColor: "#a371f7",
                    pointRadius: 5,
                    pointStyle: "rect",
                    tension: 0,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false // Using custom HTML legends
                },
                tooltip: {
                    mode: "index",
                    intersect: false,
                    backgroundColor: "#161b22",
                    titleColor: "#f0f6fc",
                    bodyColor: "#8b949e",
                    borderColor: "rgba(255, 255, 255, 0.1)",
                    borderWidth: 1,
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || "";
                            if (label) {
                                label += ": ";
                            }
                            if (context.parsed.y !== null) {
                                label += formatCurrency(context.parsed.y, currency);
                            }
                            return label;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        color: "rgba(255, 255, 255, 0.03)"
                    },
                    ticks: {
                        color: "#8b949e",
                        font: {
                            size: 11
                        }
                    }
                },
                y: {
                    grid: {
                        color: "rgba(255, 255, 255, 0.05)"
                    },
                    ticks: {
                        color: "#8b949e",
                        font: {
                            size: 11
                        },
                        callback: function(value) {
                            if (value >= 1000000) {
                                return (value / 1000000).toFixed(1) + "M";
                            }
                            return formatCurrency(value, currency).replace("VNĐ", "").trim();
                        }
                    }
                }
            }
        }
    });
}

// Trigger real-time prediction
async function triggerPredict() {
    if (!currentTicker) return;
    
    const overlay = document.getElementById("loading-overlay");
    overlay.classList.add("active");
    
    try {
        const response = await fetch(`${API_BASE}/predict/trigger/${currentTicker}`, {
            method: "POST"
        });
        
        if (!response.ok) {
            const data = await response.json();
            throw new Error(data.detail || "Không thể khởi động tiến trình dự báo");
        }
        
        // Refresh dashboard data
        await refreshDashboard();
        
        // Reload sidebar price/percent metrics
        await loadWatchlist();
        
    } catch (err) {
        alert(`Lỗi: ${err.message}`);
    } finally {
        overlay.classList.remove("active");
    }
}
