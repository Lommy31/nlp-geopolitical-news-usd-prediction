# Preprocessing rules

## 1. Use JISDOR dates as the master calendar

One row represents one fixing day. Calculate all price features (returns, MA, RSI, MACD, and so on) on this business-day index, rather than on a calendar-day index.

## 2. Define the label using the next fixing

`r = log(J_next / J_t)`, where `J_next` is the next available fixing. For Friday, the target covers Friday → Monday. That return ending on Monday spans three calendar days, so its variance is higher. There are two ways to handle this:

- Add a `gap_days` feature (1, 3, or more for extended holidays), or
- Also report metrics by group (Monday versus other days) as a diagnostic check.

## 3. Group news into intervals between fixings, rather than calendar dates

Keep Saturday and Sunday news and assign it to the interval ending at Monday's fixing. First convert all news timestamps to WIB, then align them with the JISDOR fixing time.

## 4. Choose the experiment mode and document it in the methodology

- **Strict forecasting:** use news available up to the fixing on day t to predict `r` to the next fixing. Weekend news enters Monday's input. Because it has already contributed to Monday's fixing, its signal for Tuesday tends to be weaker (already priced in). This is an honest setup, although the signal may be small.
- **Explanatory analysis (association):** use news in the interval between fixings t and t+1 to explain `r` over that same interval. This measures association rather than prediction, so the results are much better, but must not be presented as forecasting. Since the project covers “association or prediction,” both modes can be reported with clear labels.

## 5. Handle days without news

Do not invent sentiment values or carry forward old sentiment. Apply a padding mask in attention (`key_padding_mask`, as in the earlier code) and add an `n_berita` feature. For daily aggregation without attention, use a score of 0 together with `n_berita = 0` so the model can distinguish an empty input.

## 6. Aggregate news in longer intervals, such as the interval ending on Monday

Weekend intervals contain more news, so use a mean or decay-weighted aggregation rather than a raw sum to prevent the scale from jumping.

## 7. Additional checks

- Check for duplicate dates or missing JISDOR values before calculating returns.
- Fit normalization (z-scores) on the training set only, then apply it to the validation and test sets.
