# Machine Learning - Set Y: Hands-on VAE Concepts & Latent Space

**Objective:** To understand and compute key components of a Variational Autoencoder (VAE) using basic numerical operations. In this exercise, you will explore the reparameterization trick, calculate loss components (Reconstruction and KL Divergence), simulate latent variable collapse, and consider the role of the $\beta$-VAE.

**Tools:** You can use your preferred programming language (e.g., Python with NumPy/SciPy, MATLAB, Julia). You will need basic array/matrix operations, random number generation (standard normal), and math functions (exp, log, sum, mean). No dedicated ML frameworks are required.

**Setup:** We will work with a 2D input space and a 2D latent space ($d=2$). We will simulate the Encoder and Decoder using simple affine transformations (matrix multiplication + bias addition). Let $N=1000$ be the number of data points.

---

### **Notation**

*   $\bm{x} \in \mathbb{R}^2$: Input data vector. `$X_true$` is the set of $N$input vectors $\{\bm{x}_1, ..., \bm{x}_N\}$.
*   $\bm{\mu} \in \mathbb{R}^2$: Mean vector of the latent distribution $q(z|x)$.
*   $\bm{\sigma} \in \mathbb{R}^2$: Standard deviation vector of the latent distribution $q(z|x)$.
*   $\log \bm{\sigma}^2 \in \mathbb{R}^2$: Log-variance vector (element-wise log of variance).
*   $\bm{z} \in \mathbb{R}^2$: Latent space vector. `$Z$` is the set of $N$latent vectors $\{\bm{z}_1, ..., \bm{z}_N\}$.
*   $\bm{\epsilon} \in \mathbb{R}^2$: Random noise vector sampled from the standard normal distribution $\mathcal{N}(\bm{0}, I)$.
*   $\bm{x}' \in \mathbb{R}^2$: Reconstructed data vector. `$X_prime$` is the set of $N$reconstructed vectors.
*   $\mathcal{N}(\bm{\mu}, \Sigma)$: A multivariate Gaussian distribution with mean $\bm{\mu}$and covariance $\Sigma$.
*   $\mathbf{W}_{\cdot}$: Weight matrices (all are 2x2).
*   $\bm{b}_{\cdot}$: Bias vectors (all are 2D).
*   `@`: Matrix multiplication (e.g., `W @ x`).
*   `odot`: Element-wise (Hadamard) product.
*   $D_{KL}(q || p)$: The Kullback-Leibler (KL) divergence between distributions $q$ and $p$.

---

## 1. Data Generation

a. Generate $N=1000$ data points, forming the dataset `$X_true$`. Each point $\bm{x}_i$ should be sampled from a 2D Gaussian distribution $\mathcal{N}(\bm{\mu}_{data}, \Sigma_{data})$ with:
   $$
   \bm{\mu}_{data} = \begin{pmatrix} 3.0 \\ 2.0 \end{pmatrix}, \quad \Sigma_{data} = \begin{pmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \end{pmatrix}
   $$
   Store these points in an $N \times 2$ array or matrix.

b. *Optional Visualization:* Create a scatter plot of your generated data points `$X_true$`.

## 2. Simulated VAE Parameters (Scenario 1: Standard)

We will simulate a VAE's Encoder and Decoder with the following fixed parameters.

*   **Encoder:** The encoder maps an input $\bm{x}$ to the parameters of a latent distribution.
    -   Mean: $\bm{\mu} = \mathbf{W}_{\mu} \bm{x} + \bm{b}_{\mu}$
    -   Log-variance: $\log \bm{\sigma}^2 = \mathbf{W}_{ls} \bm{x} + \bm{b}_{ls}$
    -   Parameters:
        $$
        \mathbf{W}_{\mu} = \begin{pmatrix} 0.5 & 0.0 \\ 0.0 & 0.5 \end{pmatrix}, \quad \bm{b}_{\mu} = \begin{pmatrix} -0.5 \\ -0.25 \end{pmatrix}
        $$
        $$
        \mathbf{W}_{ls} = \begin{pmatrix} 0.1 & 0.0 \\ 0.0 & 0.1 \end{pmatrix}, \quad \bm{b}_{ls} = \begin{pmatrix} -1.0 \\ -1.0 \end{pmatrix}
        $$

*   **Decoder:** The decoder maps a latent vector $\bm{z}$ back to a reconstructed data point $\bm{x}'$.
    -   Reconstruction: $\bm{x}' = \mathbf{W}_{dec} \bm{z} + \bm{b}_{dec}$
    -   Parameters:
        $$
        \mathbf{W}_{dec} = \begin{pmatrix} 1.4 & 0.0 \\ 0.0 & 1.4 \end{pmatrix}, \quad \bm{b}_{dec} = \begin{pmatrix} 1.5 \\ 1.0 \end{pmatrix}
        $$

> *Note: These parameters are fixed for this exercise. In a real VAE, these would be learned during training.*

## 3. VAE Forward Pass & Loss Calculation (Scenario 1)

Perform the following calculations for each data point $\bm{x}_i$ in `$X_true$` using the parameters from Exercise 2.

a. **Encoding:** For each $\bm{x}_i$, calculate the mean vector $\bm{\mu}_i$ and the log-variance vector $\log \bm{\sigma}^2_i$.

b. **Reparameterization Trick:**
   i.  Calculate the standard deviation vectors $\bm{\sigma}_i$ using the formula:
       $\bm{\sigma}_i = \exp(0.5 \odot \log \bm{\sigma}^2_i)$

   ii. Sample $N=1000$ random noise vectors, $\{\bm{\epsilon}_1, ..., \bm{\epsilon}_N\}$, where each $\bm{\epsilon}_i \sim \mathcal{N}(\bm{0}, I)$.

   iii. Compute the set of latent vectors $Z = \{\bm{z}_1, ..., \bm{z}_N\}$ using the reparameterization formula:$\bm{z}_i = \bm{\mu}_i + \bm{\sigma}_i \odot \bm{\epsilon}_i$
   *Optional Visualization:* Create a scatter plot of the latent vectors $Z$.

c. **Decoding:** Calculate the reconstructed data points $X_{prime} = \{\bm{x}'_1, ..., \bm{x}'_N\}$ by passing the latent vectors $Z$ through the decoder.
   *Optional Visualization:* Create a scatter plot of the reconstructed points $X_{prime}$. Overlay it on the plot of $X_{true}$ if possible.

d. **Loss Calculation:**
   i.  Calculate the **Reconstruction Loss**. We use the average Mean Squared Error (MSE):

$$L_{recon} = \frac{1}{N} \sum_{i=1}^{N} ||\bm{x}_i - \bm{x}'_i||^2$$
      
where $||\bm{v}||^2 = v_1^2 + v_2^2$.

   ii. Calculate the **KL Divergence** for each point with respect to the standard normal prior $\mathcal{N}(\bm{0}, I)$. The formula for a diagonal Gaussian is:

$$
D_{KL,i} = \frac{1}{2} \sum_{j=1}^{2} (\sigma_{i,j}^2 + \mu_{i,j}^2 - \log(\sigma_{i,j}^2) - 1)
$$

Remember that $\sigma_{i,j}^2 = \exp((\log \bm{\sigma}^2_i)_j)$.
   iii. Calculate the average KL Divergence over all points:

$$
L_{KL} = \frac{1}{N} \sum_{i=1}^{N} D_{KL,i}
$$
       
   iv. Calculate the total **VAE Loss** (this is the negative ELBO, with $\beta=1$):
$$
Loss = L_{recon} + L_{KL}
$$

e. **Report your calculated values for `$L_{recon}$`, `$L_{KL}$`, and `Loss`.**

## 4. Simulating Latent Collapse (Scenario 2)

Now, we simulate a scenario where the encoder is "broken" and prioritizes matching the prior $\mathcal{N}(\bm{0}, I)$ too strongly, ignoring the input data. We will use new encoder parameters, but keep the **original data** `$X_{true}$` and the **original decoder** parameters from Exercise 2.

*   **New Encoder Parameters ('Collapse'):**
    $$
    \mathbf{W}_{\mu, c} = \begin{pmatrix} 0.01 & 0.01 \\ -0.01 & -0.01 \end{pmatrix}, \quad \bm{b}_{\mu, c} = \begin{pmatrix} 0.05 \\ -0.05 \end{pmatrix}
    $$
    $$
    \mathbf{W}_{ls, c} = \begin{pmatrix} 0.0 & 0.0 \\ 0.0 & 0.0 \end{pmatrix}, \quad \bm{b}_{ls, c} = \begin{pmatrix} -0.1 \\ -0.1 \end{pmatrix}
    $$

a. **Repeat the full forward pass and loss calculation from Exercise 3 (steps a-d)**, but using these new 'collapse' encoder parameters instead of the original ones.

b. **Report the new values for $L_{recon, c}$, $L_{KL, c}$, and $Loss_c$.**

c. *Optional Visualization:* Create a scatter plot of the new latent vectors $Z_c$. How does it compare to the latent space plot from Scenario 1?

d. **Analysis:** Compare the loss components ($L_{recon}$ vs $L_{recon, c}$ and $L_{KL}$ vs $L_{KL, c}$) between Scenario 1 and Scenario 2. Describe what happened to the latent space ($Z_c$) and the reconstruction quality ($X'_{prime}$) in Scenario 2. Why is this phenomenon, known as "latent collapse," undesirable for a generative model?

## 5. Introducing $\beta$-VAE

The standard VAE loss is $L = L_{recon} + L_{KL}$. The $\beta$-VAE modifies this to $L = L_{recon} + \beta \cdot L_{KL}$, which allows us to control the emphasis on the KL regularization term.

a. Using your results from **Scenario 1** (Exercise 3), calculate what the total loss would have been if $\beta = 0.2$. Let this be $Loss_{\beta=0.2}$.

b. Using your results from **Scenario 2** (Exercise 4), calculate what the total loss would have been if $\beta = 0.2$. Let this be $Loss_{c, \beta=0.2}$.

c. **Discussion:** In a real training process that minimizes total loss, explain how setting $\beta < 1$ (like $\beta=0.2$) might discourage the model from falling into the 'latent collapse' observed in Scenario 2. What is the potential trade-off when using $\beta < 1$?

> **Deeper Dive:** While we use $\beta < 1$ here to illustrate a point about preventing collapse, in research $\beta$-VAEs are often used with $\beta > 1$. This stronger pressure to match the prior can encourage the model to learn a *disentangled* latent space, where each latent dimension corresponds to a single, interpretable factor of variation in the data. This is a powerful technique for building more understandable generative models.