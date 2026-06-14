class MacroAgent:
    """
    Macro Analysis Agent: Evaluates global and local macroeconomic signals
    such as market volatility (VIX), bond yields, exchange rates (USDVND),
    and benchmark index returns to determine the broad market risk environment.
    """
    def __init__(self):
        pass

    def analyze(self, ticker: str, vix: float, bond_yield: float, usdvnd_change: float, index_return: float) -> str:
        report = f"--- Macro Report for {ticker} ---\n"
        report += f"1. Global & Local Indicators:\n"
        report += f"   - VIX (Volatility Index): {vix:.2f} (Fear gauge: >20 is High, <15 is Low/Risk-On)\n"
        report += f"   - US 10Y Treasury Yield: {bond_yield:.4f}% (High or rising yields pressure growth equities)\n"
        report += f"   - USD/VND Daily Change: {usdvnd_change:+.4f}% (Rising USD pressures VN Stock Market)\n"
        report += f"   - Benchmark Index Return (VNINDEX/NASDAQ): {index_return:+.4f}%\n"

        # Evaluate risk regime
        regime = "Normal"
        threats = []
        if vix > 22.0:
            regime = "High Risk / Panic"
            threats.append("Extreme global market volatility (VIX)")
        if bond_yield > 4.5:
            threats.append("High US interest rates (10Y Bond Yield)")
        if usdvnd_change > 0.0030:
            threats.append("USD strengthening / VND depreciation pressure")
            
        report += f"2. Macro Risk Regime: {regime}\n"
        if threats:
            report += f"   - Active Risks: {', '.join(threats)}\n"
        else:
            report += f"   - Active Risks: None (Stable Macro Environment)\n"

        # Determine macro bias
        if vix < 15.0 and index_return > 0:
            bias = "Bullish / Risk-On"
        elif vix > 20.0 or index_return < -0.01:
            bias = "Bearish / Risk-Off"
        else:
            bias = "Neutral / Cautious"
            
        report += f"3. Macro Bias: {bias}\n"
        return report
