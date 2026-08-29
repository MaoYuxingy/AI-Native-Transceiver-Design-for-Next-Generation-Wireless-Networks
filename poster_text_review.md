# 需要你自己重写的句子

结构性的事我已经做完了（见文末"我已经改的"）。这份文件只剩**必须出自你笔下的部分**。

导师的规则：初稿你写，AI 只改语法和拼写，不动结构、不动意思、不动专业词汇。他明确说过**中英文混着写也没关系**——所以下面每一条你都可以先用中文写清楚意思，我再帮你换成英文，不改你的表达方式。

按顺序往下做即可。**System Model 那张卡片不用动**，它的文字是干净的，可以当语气标尺。

---

## 1. Background and Motivation ← 问题最集中

### 1a. `chain` —— 导师亲自点名的

现在的三句话里用了三次 `chain`：

> A conventional transceiver is built as **a chain of blocks**, each designed on its own. The **chain** performs well, but only as well as the assumptions that link one block to the next. Deep learning suggests two ways around this: replace the whole **chain** with an autoencoder [1], or replace channel estimation and detection with a learned receiver [2].

导师给了正确说法：**conventional transceiver design**。

我没有直接替换，因为整段是围绕 "chain" 这个名词搭起来的，硬换会得到别扭的英文。**这两三句请你重写**，用你自己理解的方式说明：传统收发机是分模块设计的，各模块单独最优不等于整体最优。

### 1b. 提问要具体

> Two objections are usually raised, and both are fair:

导师说 `Two questions follow` 这类表达不清晰，要改成**具体的学术提问**，他给的例子是：

> "How much is end-to-end learning worth compared to a classical receiver...?"

请你写出这个项目实际在问的两个问题。

### 1c. 段末的金句

> We address both by measurement.

太短、太像格言。删掉，或者并进前一句。

---

## 2. Objectives and Contributions

### 2a. 第三条不是有效的 problem formulation ← 导师明确指出

现在是：

> Account for everything the learned system uses. The transmitter never sees the channel. Pilots are charged against the code rate as R(L−P)/L, and estimation error enters the effective noise as N₀(1+1/P)/|ĥ|².

**问题：公式里每个符号都没有定义。** 导师要求逐个说明 R、L、P、N₀、ĥ 分别代表什么。

这一条尤其得你写——符号定义写错就是硬伤，而我不熟悉你们领域的惯用记法。先写成这样都行：

```
R = 码率
L = 每个 block 的符号数
P = 其中导频符号数
N0 = 噪声功率谱密度
ĥ = 估计出来的信道系数
```

我再帮你把中文换成英文、排进版面。

### 2b. 你自造的词

> Explain each failure by **intervention**

`intervention` 是我造的说法。如果展示时有人问你"什么叫 intervention"，你答得上来就留着；否则换成你会怎么描述这个做法。

---

## 3. Where the AWGN Gain Comes From

> ...so the receiver cannot account for the gain. **It comes from the transmitter instead.** The learned constellation moves off the regular lattice and gives more power to the symbols that were **easiest to confuse**.

- `It comes from the transmitter instead.` —— 又是段末金句，并进前一句。
- `easiest to confuse` —— 对谁而言容易混淆？意思含混，请用你的说法。

---

## 4. Why AWGN Weights Fail on Fading

> ...the BER curve rises with SNR. **That cannot be physical**, so we held every weight fixed...

> Clamping the noise reported to the demapper back into its training range **made the curve fall properly again**.

- `That cannot be physical` —— 英文别扭。你的意思是这条曲线不可能是真实的物理行为。
- `properly` —— 含混。曲线到底怎么了？（恢复单调下降？）用具体说法。

---

## 5. Learning Without a Channel Model

> The transmitter perturbs its own symbols, **gets back** a single scalar per codeword, and updates from that alone [4], **so the channel is only ever sampled**.

> ...so reinforcement learning arrives at **much the same solution** as the gradient method.

- `gets back` —— 口语化，和全文语气不一致。
- `so the channel is only ever sampled` —— 段末金句。
- `much the same solution` —— 一张全是精确数字的海报上出现"差不多一样"，显得含糊。

---

## 6. Conclusion

> The gain from end-to-end learning is real but small on AWGN, and **we could measure none at all** once the channel fades and CSI is estimated. **Accuracy alone is therefore a weak argument** for an AI-native transceiver. **The stronger one is** that the design needs no channel model...

- `we could measure none at all` —— 语法别扭。
- `weak argument / The stronger one is` —— 这个"不是 X 而是 Y"的对比是全文第六次用。这一处的对比是真的论点所在，值得保留，但请用你自己说话的方式写出来。

---

## 7. Next Steps ← 还缺东西

### 7a. 措辞

- `might finally earn its extra parameters` —— 俏皮，不学术。
- `attacks the cause found above` —— 戏剧化。

### 7b. 导师给了具体方向：硬件部署

他建议加 1–2 句，提到**把这套设计部署到真实硬件设备上（embed into a real hardware device）**。这条目前海报上完全没有，请你补。

### 7c. 参考文献

导师要 3–4 篇。目前 Next Steps 只引了 `[3]`。OFDM/多径那条和 MMSE 均衡那条都需要出处。

**必须是你真正读过的**——展示时会有人问。

---

## 8. 两处小地方

- **Performance Analysis 新增的那条**：`Leaving the (1+1/P) factor out of the effective noise understates it by 1.76 dB at P = 2.` 这句是我从 Objectives 整句搬过来的，只把 "that last factor" 改成了 "(1+1/P) factor"（因为原来的指代对象留在另一张卡片了）。**请确认这句在新位置读起来对**。
- **`sysfig.svg` 的替代文字**里写着 "Signal chain" —— 同一个 `chain`。不显示在海报上，但要不要一起改，你定。

---

## 我已经改的（你不用管）

| 导师的意见 | 处理 |
|---|---|
| 贡献部分不要写具体数值 | `1.76 dB` 已从 Objectives 移到 Performance Analysis |
| 图表放大 | 右栏三张图从 90% 提到 100%（左栏本来就是 100%） |
| 删除图表周围的冗余说明 | `both channels, both receivers...` 那句在换成 Table 1 时已经删掉；其余图旁的文字都是实质内容，不属于该删的那一类 |
| conclusion 要有 future plan | 已是第 9 张 Next Steps 卡片 |
| 参考文献放到左下留白 | 已完成 |

---

## 交回来之后

你把重写的句子发我（中英文混着写都行），我只做两件事：

1. 改语法和拼写错误
2. 指出哪一句对不了解这个项目的读者不够清楚

**不改你的结构、意思和专业用词。** 然后我把文字排进海报，重新验证仍是单页 A1。
