# Trainable Mapping with Different Training SNR Ranges

This script compares three versions of the same trainable-constellation communication system. The only difference is the training Eb/N0 range:

- `Uniform(0, 4) dB`
- `Uniform(4, 8) dB`
- `Uniform(8, 12) dB`

## Shared communication chain

```text
1500 random bits
→ group every 6 bits
→ Sionna Mapper with 64 trainable constellation points
→ 250 complex data symbols
→ prepend 4 fixed pilots (1+0j)
→ Rayleigh block fading + AWGN
→ LS channel estimation from pilots
→ regularized ZF equalization
→ neural demapper: [Re(y_eq), Im(y_eq), log10(N0_eq)]
→ 6 bit logits for each received symbol
→ recover 1500 bits
```

## Code structure

- **Lines 51–134:** random seeds, communication parameters, training ranges and evaluation settings.
- **Lines 162–175:** bit grouping and symbol-index conversion used for SER calculation.
- **Lines 183–225:** trainable 64-point constellation and Sionna Mapper.
- **Lines 233–264:** neural bit-wise demapper (`3 → 128 → 128 → 6`).
- **Lines 272–339:** Rayleigh channel, four pilots, LS estimation and ZF equalization.
- **Lines 347–406:** complete end-to-end data flow and BCE loss.
- **Lines 435–497:** training loop for the three SNR ranges.
- **Lines 505–616:** common evaluation over `0–20 dB` with BER, SER and BLER.
- **Lines 624–890:** saving CSV files, parameters, figures and learned constellations.
- **Lines 898–972:** main experiment sequence.
