Liquidity Issuance Logic

Variables:
F_t: forwardLiquidity at time t.
R_t: resurrectedLiquidity at time t.
L_max: global issuance cap.
mu_t: expectedFlow baseline.
sigma_t: volatility tracker.
Entropy e in [0,1).
Gamma (γ): forward leverage scalar (1e18 WAD).
Lambda (λ): per-second retention factor (0<λ<=1e18).
Modulus M: entropy modulus.
Beta (β): L_max scaler (1e18 WAD).
O_p: opportunity value for failed execution.

Forward Accrual:
1. Entropy adjustment e = H(blockhash||sender) mod ENTROPY_MOD / 1e18.
2. Effective expectation E = mu_t * (1 + e).
3. Increment ΔF = γ * E / 1e18.
4. F_t+ = min(F_t + ΔF, L_max).

Resurrection:
1. Failure record stores (timestamp τ, value O_p).
2. Entropy sample r = H(blockhash||msg.sender||gasleft||execId||O_p) mod M.
3. Decay weight d = λ^{Δt}, Δt = now - τ.
4. Raw resurrectable L_r = (r * d) / 1e18.
5. Clamp against credited forward supply: L_r' = max(L_r - F_t, 0).
6. Anti-gaming: L_r' <= MAX_RESURRECTION and L_r' < F_t + O_p.
7. R_t+ = min(R_t + L_r', L_max - F_t).

Total Liquidity:
L_total = F_t + R_t, enforced <= L_max during every sync.

Caps and Sinks:
- L_max = β * maxObservedOpportunity / 1e18.
- Successful executions reduce F_t by spend value until zero.
- No retroactive mint: only entropy accrual or resurrected path adds supply.
- Cap tightening requires totalLiquidity <= new L_max, else revert.

Steady-State Considerations:
- Forward accrual saturates when ΔF pushes F_t to L_max before discounting.
- Resurrection relies on λ decay; as Δt grows, d -> 0, forcing R_t contributions to vanish.
- Combining ΔF inflows and execution sinks yields bounded oscillation around L_max under continuous flow assumptions.
