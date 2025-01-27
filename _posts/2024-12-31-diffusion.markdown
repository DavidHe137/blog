---
layout: post
title: "Diffusion Models"
date: 2024-12-31 5:00:00 -0400
categories: jekyll update
---

<!-- # Why another post on Diffusion

There are many excellent blog post explainers on Diffusion Models. Even with these resources, I found it difficult to understand the motivation for each equation, as well as how the connection between the math and the code.

This is the guide I wish I had when I learned the topic for the first time. The thought process here roughly follows the order of DDPM, so you can follow along with the paper if you'd like. -->

For now, this will be more of an annotated walkthrough of the hard parts of the DDPM paper. I'll revise more this week.

# **Definitions**

![Diffusion Process](/assets/images/diffusion/DDPM.png)

Like many other generative models, diffusion models learn a (possibly complex) target distribution by mapping a simple distribution to the target distribution.

From DDPM

> Diffusion models [53] are latent variable models of the form:
> $$p_\theta(x_0) := \int p_\theta(x_{0:T})dx_{1:T}$$
> where \(x*1, \ldots, x_T\) are latents of the same dimensionality as the data \(x_0 \sim q(x_0)\). The joint distribution \(p*\theta(x*{0:T})\) is called the reverse process, and it is defined as a Markov chain with learned Gaussian transitions starting at \(p(x_T) = \mathcal{N}(x_T; 0, I)\):
> $$p*\theta(x*{0:T}) := p(x_T) \prod*{t=1}^T p*\theta(x*{t-1}|x*t) \\ p*\theta(x*{t-1}|x_t) := \mathcal{N}(x*{t-1}; \mu*\theta(x_t, t), \Sigma*\theta(x*t, t))$$
> What distinguishes diffusion models from other types of latent variable models is that the approximate posterior \(q(x*{1:T}|x_0)\), called the forward process or diffusion process, is fixed to a Markov chain that gradually adds Gaussian noise to the data according to a variance schedule \(\beta_1, \ldots, \beta_T\):
>
> $$
> q(x_{1:T}|x_0) := \prod_{t=1}^T q(x_t|x_{t-1})\\
> q(x_t|x_{t-1}) := \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_tI)
> $$

My favorite intuition for this framework comes from the introduction of [Sohl-Dickstein, 2015](https://arxiv.org/abs/1503.03585). While our target distribution may ve too complex to learn directly, we can create a flexible family of distributions by composing simple distributions (Gaussians in this case). This feels very similar to the way we stack layers in deep neural networks, composing simple functions to create a flexible family of functions.

# The Derivation

## The Loss Function

### ELBO

We want to maximize the **likelihood of the data given our model**

$$
\max_\theta \mathbb{E}_{x}[p_\theta(x)]
$$

to help with numerical stability, we'll maximize the **log likelihood**

$$
\max_\theta \log \mathbb{E}_{x}[p_\theta(x)]
$$

For our data and model, this objective is intractable.

> TODO: This equation is intractable because...

<!-- intractable (no analytical solution + numerical integration is expensive/infeasible for high dimensional data) -->

Instead, we'll derive what's commonly known as the [Evidence Lower Bound (ELBO)](https://en.wikipedia.org/wiki/Evidence_lower_bound), a tractable approximation of the log-likelihood. Since it's a lower bound on the log likelihood, maximizing the ELBO indirectly maximizes the log-likelihood.

Let \(x\) be our target variable and \(z\) be our latent variable.

$$
\begin{aligned}
    \log p(x) &= \log \int_z p(x, z) dz \\
        &= \log \int_z p(x, z) \frac{q(z|x)}{q(z|x)} dz \\
        &= \log \mathbb{E}_{q(z|x)}[\frac{p(x, z)}{q(z|x)}] \\
        &\geq \mathbb{E}_{q(z|x)}[\log \frac{p(x, z)}{q(z|x)}] \\
\end{aligned}
$$

> TODO: Explain the ELBO derivation from the motivation of importance sampling + KL terms.

<!--
$$
\begin{aligned}
    \log p(x) &= \log \int_z p(x, z) dz \textcolor{blue}{\text{ (marginalizing out latent variable z)}} \\
        &= \log \int_z p(x, z) \frac{q(z|x)}{q(z|x)} dz \textcolor{blue}{\text{ (importance sampling)}} \\
        &= \mathbb{E}_{q(z|x)}[\log p(x, z)] \textcolor{blue}{\text{ (importance sampling)}} \\
        &= \mathbb{E}_{q(z|x)}[\log p(x|z)] + \mathbb{E}_{q(z|x)}[\log p(z)] - \mathbb{E}_{q(z|x)}[\log q(z|x)] \textcolor{blue}{\text{ (multiply by 1 using variational distribution q)}} \\
\end{aligned}
$$ -->

Substituting the terms for the diffusion process defined previously, we have

$$
x = x_0, z = x_1, \ldots, x_T \\
\mathbb{E}_{x}[\log p(x_0)] \geq \mathbb{E}_{q}\bigg[\log \frac{p(x_{0:T})}{q(x_{1:T}|x_0)}\bigg]
$$

In machine learning, we often speak in terms of minimizing a loss function, so we'll take the negative of the ELBO to get the loss function.

$$
\min_\theta \mathbb{E}_{q}\bigg[-\log \frac{p(x_{0:T})}{q(x_{1:T}|x_0)}\bigg]
$$

### Variance Reduction

> TODO: I think we can already calculate this objective via Monte-Carlo estimation, albeit with high variance. But I'm not 100% sure yet.

We can rewrite the terms to reduce the variance of our estimator.

$$
\begin{aligned}
\mathbb{E}_{q}\bigg[-\log \frac{p(x_{0:T})}{q(x_{1:T}|x_0)}\bigg] &= \mathbb{E}_{q}\bigg[-\log \frac{p(x_0|x_1)p(x_1|x_2) \ldots p(x_{T-1}|x_T)p(x_T)}{q(x_1|x_0)q(x_2|x_1) \ldots q(x_T|x_{T-1})}\bigg] \\
    &= \mathbb{E}_{q}\bigg[-\log p(x_T) - \sum_{t=1}^T \log \frac{p(x_{t-1}|x_t)}{q(x_t|x_{t-1})} \bigg] \\
    &= \ldots
\end{aligned}
$$

<!-- # magic

Now that we have a tractable loss function that is fully differentiable w.r.t to our model parameters, deep learning says we can just go go go! -->

<!-- # Evaluating Likelihoods

as we mentioned before, calculating p(x) exactly is intractable. -->

<!-- # A note on the history of Diffusion -->

<!-- ## What's next?

It's super cool that you've gotten this far! I encourage you to work through the derivation yourself, as you'll find all the places where you might not fully understand everything. Feel free to leave a like if you found this helpful. -->
