Protocol State Machine

State Definitions:
1. Idle
   Variables stable; awaiting accrual or execution signal.
2. EntropyAccrual
   Trigger: accrueForwardLiquidity(sender).
   Actions: entropy sampling E=H(blockhash||sender) mod ENTROPY_MOD, effective flow F=mu_t*(1+E/1e18), delta=gamma*F/1e18, forwardLiquidity+=delta (bounded by L_max).
   Exit: emit ForwardAccrued, recompute totalLiquidity.
3. ExecutionCommit
   Trigger: recordExecution(execId,value,success).
   Actions: snapshot ExecutionRecord, branch on success flag.
   Success path: state ExecutionSuccess.
   Failure path: state FailureRecorded.
4. ExecutionSuccess
   Actions: discount forwardLiquidity by value, floor at zero, sync totals.
5. FailureRecorded
   Actions: store execution metadata, fire FailureRecorded event, move to Resurrection.
6. Resurrection
   Actions: entropy sample R=H(blockhash||msg.sender||gasleft||execId||value) mod modulus, decay factor D=lambda^Δt (fast exp), resurrectable=R*D/1e18, clamp= max(R*D/1e18 - forwardLiquidity,0), enforce clamp<=MAX_RESURRECTION and clamp<forwardLiquidity+value.
   Exit: resurrectedLiquidity += clamp, ensure forward + resurrected <= L_max, emit LiquidityResurrected.
7. CapTightening
   Trigger: updateLmax(newObs).
   Actions: L_max = beta*newObs/1e18, assert totalLiquidity<=L_max.
8. SupplySync
   Triggered after accrual/resurrection/discount.
   Actions: totalLiquidity = forwardLiquidity + resurrectedLiquidity, assert totalLiquidity<=L_max, emit TotalLiquiditySynced.

State Transitions:
Idle -> EntropyAccrual [accrueForwardLiquidity]
EntropyAccrual -> SupplySync [post accrual]
Idle -> ExecutionCommit [recordExecution]
ExecutionCommit -> ExecutionSuccess [success==true]
ExecutionCommit -> FailureRecorded [success==false]
FailureRecorded -> Resurrection [recordFailure/_resurrect]
Resurrection -> SupplySync [after resurrectedLiquidity update]
ExecutionSuccess -> SupplySync [after discount]
Idle -> CapTightening [updateLmax]
CapTightening -> Idle [cap validated]

Invariant Matrix:
- I1: forwardLiquidity <= L_max enforced in EntropyAccrual.
- I2: resurrectedLiquidity + forwardLiquidity <= L_max enforced in Resurrection and SupplySync.
- I3: clamp < forwardLiquidity + originalO ensures resurrection cannot exceed original opportunity plus outstanding forward credit.
- I4: lambda <= 1e18 validated at deployment/update.
- I5: modulus != 0 before entropy modulus operations.
- I6: MAX_RESURRECTION bounds per-incident issuance.
