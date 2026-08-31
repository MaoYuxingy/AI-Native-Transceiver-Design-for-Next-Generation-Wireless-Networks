# A 的部分 —— 逐句读稿

Yuxing 的四段，可以直接照着读。措辞跟最新一版海报一致（导师改过的那版）。

**读的时候：**
- 每个 `//` 是换气点，停半拍，不是停顿符号，别读出来。
- `[可删]` 标的句子是超时的时候先砍的，砍了不影响意思。
- 数字**照写的读**，尤其 "ten to the minus three" 这种，别读成 "one e minus three"。
- 照读容易听出来在念稿。补救办法只有一个：**先出声读三遍**，读到不用盯着字也能往下接，那时候语气就自然了。

---

## 【0:00–0:10】幻灯片 1　目标 ~10 秒

Hi, I'm Yuxing Mao, and this is Haikun Xu. //
Our project is AI-Native Transceiver Design, supervised by Doctor Huynh Nguyen. //
We use deep learning to redesign a wireless transceiver.

---

## 【0:10–0:40】幻灯片 2　目标 ~30 秒

A conventional transceiver is built from separate blocks: //
coding, modulation, channel estimation, detection. //
Each block is optimal for its own task, but not for the system as a whole, //
so the end-to-end performance is limited. //

Instead, we replace those blocks with neural networks, trained together as one autoencoder. //

That gives us two questions. //
How much is learning actually worth? //
And can the system train with no channel model at all?

---

## 【0:40–2:00】幻灯片 3　目标 ~80 秒

**这段是你的重头戏，别赶。**

The theme here is fairness. //

We built two systems, in Sionna and PyTorch. //
One has a trainable constellation and a neural demapper. //
The other is classical: sixty-four QAM, Gray labelling, and a posteriori probability demapping. //
Both share the same LDPC code, the same channel, and the same random seeds, //
and both are evaluated on the same terms. //
So any difference in bit error rate comes only from the mapper and the demapper. //

[可删] Being fair has a cost, and we pay it twice. //

First, pilots are not free. //
They take up symbol slots, so the effective code rate drops to R times L minus P, over L. //

Second, the channel estimate is not exact. //
[可删] When we equalise with an estimate, the error gets amplified. //
We put that into the effective noise, as a factor of one plus one over P. //
With only two pilots, that factor is worth **one point seven six dB**. //
And this is a term we added ourselves. //
Before we added it, we were understating our own noise. //

On measurement. //
Every point is **sixty-four thousand** codewords. //
We throw away any point with fewer than **thirty** block errors, //
because the relative standard error there is over eighteen percent, //
[可删] and noise like that gets read as structure.

---

## 【3:20–4:20】幻灯片 7　目标 ~60 秒

My skill is research rigour. //
Specifically, checking the thing that might go against you. //

Halfway through the project, we realised our effective noise model was missing a term. //
It was the one plus one over P I mentioned earlier. //
We added it, and our own numbers got worse. //

It also reversed a conclusion. //
Two pilots had looked like the best choice. //
Once the estimation error was counted properly, two pilots became the worst. //
[可删] The saving in overhead was real, but we had not been paying for the extra estimation error. //

We changed it anyway, //
and we wrote a check that tests the model against tensors the simulator produces, rather than trusting it. //

Beyond LivSURF. //
In research, or in engineering, you will always get results that happen to favour you. //
What decides whether anyone can trust them //
is whether you are willing to test the thing that could embarrass you. //
That habit carries into a PhD, and into any job that touches data.

---

## 【5:20–6:00】幻灯片 9　A 讲前半，目标 ~25 秒

What surprised me most: under fading, the gain is small. //
Zero point zero one and zero point one one dB, //
and the wrong way at ten to the minus four. //
But the real finding was somewhere else: //
training with no channel model at all costs only **zero point one two to zero point one four dB**. //
A small difference, measured cleanly, is worth as much as a big one.

（交给 Haikun 收尾）

---

# 数字怎么念

| 写的 | 念的 |
|---|---|
| 64-QAM | sixty-four QAM |
| 1e-2 / 1e-3 / 1e-4 | ten to the minus two / three / four |
| 1.76 dB | one point seven six dB |
| 0.12 dB | zero point one two dB |
| Eb/N0 | E b over N naught |
| R(L−P)/L | R times L minus P, over L |
| (1+1/P) | one plus one over P |
| APP | a posteriori probability |

# 超时怎么办

先砍 `[可删]`，两句省 8 秒左右。还超，就把 0:40–2:00 里 "On measurement" 那段压成一句：
"Every point is sixty-four thousand codewords, and we drop any point with fewer than thirty block errors."

# 时间核对

按 140 词／分钟估的（正常语速偏快一点，读稿一般就这个速度）：

| 段落 | 分配 | 全读 | 砍掉 [可删] |
|---|---|---|---|
| 幻灯片 1 | 10s | 12s | 12s |
| 幻灯片 2 | 30s | 30s | 30s |
| 幻灯片 3 | 80s | 89s | **77s** |
| 幻灯片 7 | 60s | 69s | **62s** |
| 幻灯片 9 | 25s | 30s | 30s |
| **合计** | **~200s** | 231s | **211s** |

全读会超 30 秒，**砍掉 [可删] 之后基本卡得住**。

先掐着表读一遍幻灯片 3，那段最长。如果你比 140 词／分钟慢，就把 [可删] 全砍掉。
