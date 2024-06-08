# 活性化函数

人間が考えるとき、精確な数値推定を生み出すことはめったにありません。代わりに、より一般的に行われることは分類です。

垂直軸は確率に変わりました。

1は（有毒）を示し、

0は（無毒）を示します。

中間値は存在しません。

我々は、一次元の出力がある閾値より大きい場合に固定された出力1を、その閾値より小さい場合に固定された出力0を期待しています。

以前の線形関数の結果を、分段関数に入れて二次的に処理します。そして、この新たに導入された分段関数が、いわゆる**「活性化関数」**の一種です。

Rosenblattのパーセプトロンでも活性化関数が存在します。

ローゼンブラットのパーセプトロンは、線形関数の計算結果を、類似した符号関数に入れて分類します。

ロジスティック関数

$$
\begin{align}
\phi(y) = \frac{L}{1 + e^{-k(y - y_0)}} \notag \\
\end{align}
$$

標準的なロジスティック関数

$$
\begin{align}
\phi(y) = \frac{1}{1 + e^{-y}} \notag \\
\end{align}
$$

すなわち、 \( L = 1, k = 1, y_0 = 0 \)

ステップ関数の導関数はインパルス関数であり、これは勾配降下に不利です。

したがって、シグモイド関数を使用します。これは至る所で導関可能であり、導関数がゼロではありません。

入力 \( x \) から出力 \( a \) まで、実際には中間の \( z \) を介して関係が形成される複合関数です。入力 \( x \) と最終出力 \( a \) の関数グラフは、 \( w \) と \( b \) を調整することで変えることができます。この \( x \) 形の曲線は移動および伸縮することができます。そして、 \( w \) と \( b \) の調整方法は以前学んだ勾配降下アルゴリズムです。

勾配降下アルゴリズムを用いることで、最終的に曲線を変化させることができます。

予測モデル

$$
\begin{align}
z &= wx + b \notag \\
a &= \text{sigmoid}(z) \notag \\
a &= \text{sigmoid}(wx + b) \notag \\
\end{align}
$$

コスト関数

$$
\begin{align}
e &= (y - a)^2 \notag \\
e &= (y - \text{sigmoid}(wx + b))^2 \notag \\
\end{align}
$$

\( w \) と \( b \) それぞれについての導関数

$$
\begin{align}
\frac{\partial e}{\partial b}\quad , \quad \frac{\partial e}{\partial w} \notag \\
\end{align}
$$

方法1：

定義による方法

$$
\begin{align}
\frac{\partial e}{\partial w} = \frac{(y - \text{sigmoid}((w + \Delta w)x + b))^2 - (y - \text{sigmoid}(wx + b))^2}{\Delta w} \notag \\
\end{align}
$$

**連鎖法則**

$$
\begin{align}
e &= (y - \text{sigmoid}(wx + b))^2 \notag \\
\notag \\
z &= wx + b \notag \\
\frac{dz}{db} &= 1 \notag \\
\frac{dz}{dw} &= x \notag \\
\notag \\
a &= \text{sigmoid}(z) \notag \\
a &= \text{sigmoid}(wx + b) \notag \\
\frac{da}{dz} &= a(1 - a) \notag \\
\notag \\
e &= (y - a)^2 \notag \\
\frac{de}{da} &= -2(y - a) \notag \\
\notag \\
\frac{\partial e}{\partial w} &= \frac{\partial e}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w} = -2(y - a)a(1 - a)x \notag \\
\frac{\partial e}{\partial b} &= \frac{\partial e}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial b} = -2(y - a)a(1 - a) \notag \\
\end{align}
$$

多層ニューラルネットワークの本質は、非常に深く入れ子になった複合関数です。

もちろん、神経元の層数を縦に増やすだけでなく、一層あたりの神経元の数を横に増やすこともできます。複合関数の導関数を求める連鎖法則を利用し、勾配降下と逆伝播を完璧に組み合わせることで、このネットワークがどれだけ深くても広くても、同じ方法でコスト関数を計算し、各神経元の重みパラメータの導関数を求めることができます。そして、勾配降下アルゴリズムを用いてこれらのパラメータを調整します。

多層のニューラルネットワークでは、もし各神経元が線形関数であれば、非常に多くの神経元を使って複雑なニューラルネットワークを構築しても、それは数学的には依然として線形システムです。

なぜなら、線形関数はどのように重ねても結果は線形関数のままです。

一方、活性化関数は非線形であり、複雑な問題を処理する能力を持っています。

活性化関数を持つ神経元を複数のノードに拡張したネットワークを導入すると、これが一般的な場合における真の逆伝播アルゴリズムと言えます。コスト関数が神経ネットワーク内の各ノードに逆伝播されます。

パラメータがコスト関数に与える影響を計算し、それに基づいてパラメータを調整します。


