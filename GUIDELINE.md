# Hướng dẫn thực hiện dự án: Modular Trading Strategy Backtester

> Backtesting & Risk Analytics Engine cho chiến lược giao dịch hệ thống
> Tài liệu guideline chi tiết — bám theo để không miss bước nào
> Stack: Python · pandas · NumPy · yfinance · Streamlit · Plotly

---

## 1. Tổng quan dự án

Đây là một **research-oriented backtesting engine** viết bằng Python, dùng để đánh giá các chiến lược giao dịch hệ thống (systematic trading strategies) trên dữ liệu thị trường lịch sử. Dự án được xây dựng như một **mini-product có kiến trúc rõ ràng**, không phải một script đơn lẻ hay một Jupyter notebook lộn xộn.

### Mục tiêu CV

Dự án nhắm tới việc chứng minh đồng thời ba năng lực khi ứng tuyển vào finance hoặc consulting:

- **Kỹ năng lập trình & software engineering** — kiến trúc module hoá, design pattern, testing.
- **Hiểu biết tài chính định lượng** — chiến lược giao dịch, risk metrics, validation methodology.
- **Tư duy nghiên cứu & trình bày** — đặt câu hỏi nghiên cứu, để dữ liệu trả lời, trình bày trực quan.

### Research question — trái tim của dự án

Thay vì chỉ đăng lên GitHub "đây là backtester của tôi", hãy đóng khung toàn bộ dự án quanh một câu hỏi nghiên cứu:

> **Research Question:** Liệu các chiến lược technical cổ điển (trend-following, mean reversion, momentum) có tạo ra risk-adjusted returns bền vững sau khi tính đến transaction costs thực tế hay không?

Điểm mấu chốt: kể cả khi kết luận cuối cùng là *"không chiến lược nào consistently beat buy-and-hold"*, dự án vẫn thành công. Nó cho thấy bạn xây một framework, đặt hypothesis và để dữ liệu trả lời — chứ không build để chứng minh một kết luận có sẵn.

### Nguyên tắc xuyên suốt: đừng overclaim

Với dữ liệu daily miễn phí và giả định đơn giản, **tuyệt đối tránh** những câu như "Built a professional quantitative trading system." Nó nghe overclaim. Cách định vị đúng là mô tả chính xác những gì đã làm — technical, đáng tin, không bullshit.

---

## 2. Công nghệ sử dụng

| Thành phần | Công cụ | Vai trò |
|---|---|---|
| Ngôn ngữ | Python 3.10+ | Toàn bộ engine và logic |
| Xử lý dữ liệu | pandas, NumPy | Time series, tính toán vector hoá |
| Nguồn dữ liệu | yfinance | Tải dữ liệu giá lịch sử miễn phí |
| Thống kê | SciPy, statsmodels | Regression (alpha/beta), test phân phối |
| Dashboard | Streamlit | Giao diện tương tác, không cần code |
| Biểu đồ tương tác | Plotly | Equity curve, trade markers, hover/zoom |
| Biểu đồ tĩnh | Matplotlib, Seaborn | Histogram, heatmap |
| Testing | pytest | Unit test cho engine và analytics |

**Lưu ý về AI:** Dự án này KHÔNG cần LLM chỉ để "cho có AI". Core value phải là: data → trading rules → portfolio simulation → risk measurement → validation. AI (ví dụ: "giải thích vì sao chiến lược underperform giai đoạn này") chỉ nên là bonus feature thêm vào sau cùng, không phải trọng tâm.

---

## 3. Kiến trúc & cấu trúc thư mục

Cấu trúc dạng module giúp recruiter/technical interviewer nhìn GitHub thấy ngay bạn biết **software engineering**, chứ không chỉ biết chạy phân tích trong notebook.

```
trading-backtester/
│
├── data/
│   └── market_data.py        # Tải & làm sạch dữ liệu giá (yfinance)
│
├── strategies/
│   ├── base_strategy.py      # Interface/abstract class chung
│   ├── moving_average.py     # MA crossover (trend-following)
│   ├── mean_reversion.py     # RSI mean reversion
│   └── momentum.py           # Momentum
│
├── engine/
│   ├── portfolio.py          # Theo dõi tiền mặt & vị thế
│   ├── broker.py             # Khớp lệnh, áp transaction cost
│   └── backtester.py         # Vòng lặp backtest chính
│
├── analytics/
│   ├── metrics.py            # Return, Sharpe, Sortino, Calmar...
│   ├── risk.py               # Volatility, drawdown, VaR, CVaR
│   └── validation.py         # Walk-forward, in/out-of-sample
│
├── visualization/
│   └── charts.py             # Plotly/Seaborn charts
│
├── tests/                    # pytest unit tests
│
├── app.py                    # Streamlit dashboard (entry point)
└── README.md                 # Research report
```

**Nguyên tắc kiến trúc quan trọng:** UI chỉ thu thập một "config object" rồi truyền vào engine. Engine KHÔNG biết gì về Streamlit. Nhờ vậy backtester dùng được cả qua code lẫn qua UI — một điểm software engineering đáng khoe khi phỏng vấn.

---

## 4. Lộ trình thực hiện theo Phase

Làm **tuần tự** theo thứ tự này. Đừng nhảy sang portfolio-level backtesting cho đến khi Phase 1–5 đã sạch. Mỗi phase nên là một mốc commit rõ ràng trên Git.

### Phase 1 — Minimum Viable Backtester

**Mục tiêu:** một backtester chạy được end-to-end với một chiến lược đơn giản nhất, trên một mã cổ phiếu.

Các bước:
1. Viết module data ingestion: tải dữ liệu OHLCV từ yfinance theo ticker và khoảng thời gian; xử lý ngày thiếu, dữ liệu rỗng.
2. Xây Portfolio class: theo dõi tiền mặt, số cổ phiếu đang giữ, tổng giá trị danh mục theo thời gian.
3. Xây Broker: nhận lệnh mua/bán, cập nhật portfolio (chưa cần transaction cost ở bước này, thêm ở Phase 4).
4. Xây vòng lặp backtester: đi qua dữ liệu theo từng ngày, giả định không biết tương lai (no look-ahead bias).
5. Cài đặt benchmark buy-and-hold để so sánh.
6. Ghi lại trade log: mỗi lệnh gồm ngày, loại lệnh, giá, khối lượng.

**Định nghĩa "Done" cho Phase 1:** Chạy được một backtest hoàn chỉnh với chiến lược giả (ví dụ: mua ngày đầu, giữ đến cuối) và in ra giá trị danh mục cuối kỳ khớp với buy-and-hold. Đây là bằng chứng engine không có lỗi logic cơ bản.

### Phase 2 — Chiến lược giao dịch

**Mục tiêu:** cài đặt các chiến lược thật, dưới dạng module có thể swap qua một interface chung (strategy pattern).

Các chiến lược:
- **Moving Average Crossover** — mua khi MA ngắn cắt lên trên MA dài (golden cross), bán khi cắt xuống (death cross). Đại diện cho trend-following.
- **RSI Mean Reversion** — mua khi RSI dưới ngưỡng oversold (vd 30), bán khi trên ngưỡng overbought (vd 70). Triết lý ngược với trend-following: giá overshoot rồi quay đầu.
- **Momentum** — mua tài sản có hiệu suất tốt nhất N tháng gần nhất, tái cân bằng định kỳ. Một trong những market anomaly được ghi nhận nhiều nhất trong academic finance.

Yêu cầu kỹ thuật:
1. Định nghĩa base_strategy với một method chuẩn, ví dụ generate_signals(data) trả về chuỗi tín hiệu mua/bán/giữ.
2. Mỗi chiến lược kế thừa base và cài đặt riêng logic, KHÔNG đụng vào engine.
3. Chiến lược nhận tham số qua constructor (vd fast/slow window) để về sau UI truyền vào được.

### Phase 3 — Performance Analytics

**Mục tiêu:** đo hiệu suất bằng bộ chỉ số chuyên nghiệp, nhóm theo return / risk / risk-adjusted / trade-level.

**Return metrics:**
- Total return — tổng % lãi/lỗ toàn kỳ.
- Annualized return (CAGR) — quy về mức trung bình mỗi năm, cho phép so sánh công bằng giữa các kỳ khác độ dài.
- Return vs benchmark — chênh lệch so với buy-and-hold.

**Risk metrics:**
- Volatility (annualized) — std của daily returns × √252.
- Maximum drawdown — mức sụt sâu nhất từ đỉnh xuống đáy.
- Drawdown duration — thời gian phục hồi về đỉnh cũ.
- VaR (95%) — historical & parametric. Trình bày cạnh nhau để minh hoạ parametric đánh giá thấp rủi ro do fat tails.
- CVaR / Expected Shortfall — trung bình khoản lỗ trong phần đuôi vượt VaR.

**Risk-adjusted metrics:**
- Sharpe ratio — return trên mỗi đơn vị rủi ro tổng. Chỉ số dân finance quan tâm nhất.
- Sortino ratio — chỉ phạt downside deviation, không phạt biến động phía tăng.
- Calmar ratio — annualized return chia max drawdown.

**Distribution & market metrics (thể hiện nền tảng toán):**
- Skewness, Kurtosis — bằng chứng định lượng của fat tails; đi kèm Jarque-Bera test (kiểm định tính chuẩn).
- Alpha, Beta, R² — từ hồi quy OLS returns chiến lược lên returns thị trường (CAPM).

**Trade-level statistics:**
- Number of trades — liên hệ trực tiếp với transaction cost.
- Win rate — % lệnh có lãi (kèm cảnh báo: win rate cao ≠ chiến lược tốt).
- Average win / average loss, Profit factor.

**Điểm nói khi phỏng vấn:** Điều tạo khác biệt không phải TÍNH được các chỉ số (thư viện làm hết) mà là hiểu GIẢ ĐỊNH đằng sau chúng. Ví dụ: Sharpe giả định returns phân phối chuẩn, nhưng skewness/kurtosis bạn tính chứng minh giả định đó sai — nên Sharpe đánh giá quá cao các chiến lược có đuôi trái dày. "Tôi tính Sharpe, nhưng cũng tính kurtosis để biết khi nào KHÔNG nên tin Sharpe."

### Phase 4 — Robustness & Validation

**Mục tiêu:** chứng minh chiến lược không phải là kết quả của overfitting. Đây là phần nâng dự án từ "solid" lên "nổi bật".

**4.1 — Transaction cost scenarios:** Cho phép thay đổi commission, bid-ask spread, slippage, rồi chạy scenario để thấy edge biến mất thế nào sau chi phí:

| Transaction cost | Strategy return | Sharpe |
|---|---|---|
| 0% | 28% | 1.45 |
| 0.05% | 21% | 1.18 |
| 0.10% | 14% | 0.82 |
| 0.25% | 2% | 0.21 |

Insight: một chiến lược có thể có edge rõ ràng trước chi phí nhưng không còn edge kinh tế ý nghĩa nào sau market frictions thực tế.

**4.2 — Parameter sensitivity analysis:** Quét qua dải tham số (vd fast MA 10→100, slow MA 50→300) và tạo heatmap Sharpe. Trọng tâm KHÔNG phải tìm Sharpe cao nhất, mà kiểm tra tính robust:
- Robust — 48/190 → 0.91 ; 50/200 → 0.93 ; 55/210 → 0.89 (ổn định quanh vùng lân cận).
- Nghi overfit — 49/197 → 1.82 nhưng mọi tổ hợp khác ≈ 0.3 (một đỉnh cô lập).

**4.3 — In-sample / Out-of-sample & Walk-forward:**

> **Lưu ý technical wording:** Vì các chiến lược là RULE-BASED (không phải machine learning), dùng "in-sample / out-of-sample" thay vì "train/test". In-sample = chọn/đánh giá parameters; out-of-sample = kiểm định. Chính xác hơn khi giải thích với người biết quant.

Walk-forward: lặp lại quá trình theo thời gian, cuốn cửa sổ về phía trước:

| In-sample (chọn parameter) | Out-of-sample (test) |
|---|---|
| 2015–2017 | 2018 |
| 2016–2018 | 2019 |
| 2017–2019 | 2020 |

Ý nghĩa: một chiến lược đẹp trên dữ liệu lịch sử có thể chỉ đang overfit vào nhiễu. Walk-forward cho thấy bạn hiểu vấn đề cốt lõi này của quantitative finance.

### Phase 5 — Interactive Dashboard & Visualization

**Mục tiêu:** người dùng test chiến lược có sẵn hoàn toàn qua giao diện, không đụng vào code.

**5.1 — Các input cần expose ra UI (thường ở sidebar):**
- Data selection: ticker (chọn 1 hoặc nhiều mã), date range picker, tần suất dữ liệu, initial capital.
- Strategy selection: dropdown chọn chiến lược. Quan trọng — parameter hiển thị phải THAY ĐỔI ĐỘNG theo chiến lược được chọn, không đổ hết ra cùng lúc.
- Parameter theo chiến lược:
  - MA Crossover: fast MA window, slow MA window (slider, ràng buộc fast < slow).
  - RSI: RSI period (vd 14), ngưỡng oversold (vd 30), ngưỡng overbought (vd 70).
  - Momentum: lookback period, rebalance frequency, số/tỷ lệ tài sản giữ.
- Market frictions: ô nhập commission, bid-ask spread, slippage — để người dùng tự chạy scenario trước/sau chi phí.
- Validation config: toggle bật/tắt walk-forward; nếu bật, chọn độ dài in-sample & out-of-sample window.
- Action: nút "Run Backtest" rõ ràng.

**5.2 — Nguyên tắc thiết kế input:**
- Input validation — chặn fast MA > slow MA, date range rỗng ngay ở UI (thể hiện nghĩ về edge case).
- Sensible defaults — mỗi ô có giá trị mặc định chuẩn (RSI 14, MA 50/200) để chạy thử ngay trong 5 giây.
- Tách config khỏi engine — UI chỉ tạo config object rồi truyền vào engine.

**5.3 — Các biểu đồ output (khu vực chính):**

Biểu đồ cốt lõi:
- Equity curve — giá trị danh mục theo thời gian, vẽ chồng với buy-and-hold benchmark (2 màu khác nhau). Biểu đồ quan trọng nhất.
- Drawdown chart — vùng tô đỏ dưới trục 0, cho thấy độ sâu sụt giảm theo thời gian.
- Trade markers — chấm điểm mua (tam giác xanh) / bán (tam giác đỏ) trên đường giá.

Biểu đồ phân tích sâu:
- Returns distribution histogram — kèm đường phân phối chuẩn chồng lên để lộ skew & fat tails.
- Parameter sensitivity heatmap — Sharpe theo cặp tham số; nhìn ra vùng robust vs đỉnh cô lập.
- Rolling metrics — Sharpe/volatility cuộn theo thời gian, cho thấy độ ổn định qua các giai đoạn.

Bảng số liệu: mỗi chiến lược một hàng cạnh benchmark, tô màu điều kiện (xanh tốt / đỏ xấu).

**5.4 — Công cụ vẽ:**
- Plotly — cho biểu đồ tương tác chính (equity curve, trade markers): hover, zoom, pan.
- Matplotlib/Seaborn — cho biểu đồ tĩnh (histogram, heatmap); Seaborn heatmap đẹp sẵn.

**"Đẹp mà không rỗng":** Mỗi biểu đồ phải TRẢ LỜI MỘT CÂU HỎI: equity curve → có thắng benchmark không; drawdown → rủi ro tệ đến đâu; distribution → returns có normal không. Khi trình bày với recruiter, gắn mỗi hình với câu hỏi nó trả lời. Nhất quán bảng màu, luôn có tiêu đề & nhãn trục, tránh nhồi quá nhiều vào một biểu đồ. Sạch sẽ dễ đọc luôn thắng hoa mỹ rối mắt.

---

## 5. Phase 6 (tùy chọn) — Portfolio-level Backtesting

Chỉ làm khi Phase 1–5 đã sạch. Thay vì chạy chiến lược trên một mã, chạy trên một rổ (AAPL, MSFT, NVDA, AMZN, GOOGL). Chiến lược tạo tín hiệu cho từng tài sản, sau đó một portfolio engine quyết định:
- Position sizing — phân bổ vốn cho mỗi vị thế (vd tối đa 20%/cổ phiếu, hoặc volatility-adjusted sizing).
- Maximum exposure — giới hạn tỷ trọng tổng.
- Rebalance frequency — tần suất tái cân bằng.
- Cash allocation — phần tiền mặt giữ lại.

**Vì sao đáng làm:** Nâng dự án từ "Trading strategy simulator" lên "Portfolio backtesting and risk analytics engine" — cái tên thứ hai nghe mạnh hơn hẳn trên CV.

---

## 6. README như một research report

README chính là nơi biến GitHub repo thành một mini research project. Cấu trúc gợi ý:

**Research Question:** Can classic technical strategies outperform buy-and-hold after accounting for realistic transaction costs?

**Methodology:**
- Historical period được test.
- Assets tested (danh sách mã).
- Transaction cost assumptions.
- Parameter ranges đã quét.
- Walk-forward methodology (mô tả cửa sổ in/out-of-sample).

**Results:**

| Strategy | Return | Sharpe | Max DD | vs Benchmark |
|---|---|---|---|---|
| Buy & Hold | — | — | — | — |
| MA Crossover | | | | |
| RSI Mean Reversion | | | | |
| Momentum | | | | |

**Key Insight (ví dụ):** "Momentum generated the highest gross returns, but performance deteriorated significantly after transaction costs. Mean reversion showed more stable results in certain market regimes but failed to consistently outperform the benchmark out-of-sample."

---

## 7. Cách trình bày trên CV

**CV line khuyên dùng:**

> Built a modular Python backtesting engine to evaluate trend-following, momentum, and mean-reversion strategies, incorporating portfolio simulation, market frictions, benchmark comparison, risk-adjusted performance analysis, and walk-forward validation.

Cụ thể, thể hiện đúng range của dự án, technical nhưng không overclaim.

**Pitch 30 giây cho recruiter không chuyên:** "Tôi xây một công cụ test các chiến lược giao dịch trên dữ liệu lịch sử — cài đặt vài chiến lược cổ điển, tính đến chi phí giao dịch thực tế, và đo chúng bằng risk-adjusted returns so với việc chỉ mua và giữ."

---

## 8. Checklist tổng — bám theo để không miss

| Phase | Hạng mục | Xong? |
|---|---|---|
| 1 | Data ingestion từ yfinance (xử lý ngày thiếu) | ☐ |
| 1 | Portfolio class (cash, positions, giá trị) | ☐ |
| 1 | Broker khớp lệnh | ☐ |
| 1 | Vòng lặp backtester (no look-ahead) | ☐ |
| 1 | Buy-and-hold benchmark | ☐ |
| 1 | Trade log | ☐ |
| 2 | base_strategy interface | ☐ |
| 2 | Moving Average Crossover | ☐ |
| 2 | RSI Mean Reversion | ☐ |
| 2 | Momentum | ☐ |
| 3 | Return metrics (total, CAGR, vs benchmark) | ☐ |
| 3 | Risk metrics (vol, drawdown, VaR, CVaR) | ☐ |
| 3 | Risk-adjusted (Sharpe, Sortino, Calmar) | ☐ |
| 3 | Distribution (skew, kurtosis, Jarque-Bera) | ☐ |
| 3 | Market (alpha, beta, R²) | ☐ |
| 3 | Trade stats (win rate, profit factor...) | ☐ |
| 4 | Transaction cost scenarios | ☐ |
| 4 | Parameter sensitivity heatmap | ☐ |
| 4 | In/out-of-sample + walk-forward | ☐ |
| 5 | Streamlit inputs (dynamic parameters) | ☐ |
| 5 | Input validation + sensible defaults | ☐ |
| 5 | Equity curve + drawdown + trade markers | ☐ |
| 5 | Distribution histogram + heatmap | ☐ |
| 5 | Bảng metrics tô màu | ☐ |
| — | pytest cho engine & analytics | ☐ |
| — | README dạng research report | ☐ |
| 6 | (Tùy chọn) Portfolio-level backtesting | ☐ |

---

*Hết — bám checklist, commit theo từng phase, và luôn hỏi "biểu đồ/chỉ số này trả lời câu hỏi gì?"*
