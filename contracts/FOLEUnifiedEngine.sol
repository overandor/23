// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "./ForwardLiquidityEngine.sol";
import "./ReactiveResurrectionEngine.sol";

/// @title FOLEUnifiedEngine
/// @notice Merges forward accrual and resurrection logic into a single issuance pipeline.
contract FOLEUnifiedEngine is ForwardLiquidityEngine, ReactiveResurrectionEngine {
    uint256 public totalLiquidity;

    event ExecutionRecorded(bytes32 indexed execId, uint256 value, bool success);
    event TotalLiquiditySynced(uint256 forwardComponent, uint256 resurrectedComponent, uint256 total);

    constructor(uint256 _gamma, uint256 _lambda, uint256 _modulus, uint256 _beta) {
        require(_lambda <= WAD, "FOLE: invalid lambda");
        require(_modulus != 0, "FOLE: invalid modulus");
        gamma = _gamma;
        lambda = _lambda;
        modulus = _modulus;
        beta = _beta;
    }

    function accrueForwardLiquidity(address sender) public virtual override {
        ForwardLiquidityEngine.accrueForwardLiquidity(sender);
        _mintLiquidity();
    }

    function recordFailure(bytes32 execId, uint256 value) public virtual override {
        super.recordFailure(execId, value);
        _mintLiquidity();
    }

    function recordExecution(bytes32 execId, uint256 value, bool success) external {
        executions[execId] = ExecutionRecord(block.timestamp, value, success);

        if (success) {
            _discountForward(value);
        } else {
            _resurrect(execId, value);
        }

        emit ExecutionRecorded(execId, value, success);
        _mintLiquidity();
    }

    function updateLmax(uint256 maxObservedOpportunity) public virtual override {
        ForwardLiquidityEngine.updateLmax(maxObservedOpportunity);
        require(totalLiquidity <= L_max, "FOLE: cap tightening violation");
    }

    function _discountForward(uint256 value) internal {
        if (forwardLiquidity >= value) {
            forwardLiquidity -= value;
        } else {
            forwardLiquidity = 0;
        }
    }

    function _mintLiquidity() internal {
        totalLiquidity = forwardLiquidity + resurrectedLiquidity;
        require(totalLiquidity <= L_max, "FOLE: global cap exceeded");
        emit TotalLiquiditySynced(forwardLiquidity, resurrectedLiquidity, totalLiquidity);
    }

    function _forwardLiquidityReference() internal view override returns (uint256) {
        return forwardLiquidity;
    }

    function _globalLiquidityCap() internal view override returns (uint256) {
        return L_max;
    }
}
