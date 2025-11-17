// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title ReactiveResurrectionEngine
/// @notice Handles execution failures and liquidity resurrection with strict decay controls.
abstract contract ReactiveResurrectionEngine {
    struct ExecutionRecord {
        uint256 timestamp;
        uint256 value;
        bool executed;
    }

    mapping(bytes32 => ExecutionRecord) public executions;

    uint256 public resurrectedLiquidity;
    uint256 public constant MAX_RESURRECTION = 1_000_000 * 1e18;
    uint256 public lambda; // Per-second retention factor in WAD.
    uint256 public modulus;

    uint256 internal constant WAD = 1e18;

    event FailureRecorded(bytes32 indexed execId, uint256 value);
    event LiquidityResurrected(bytes32 indexed execId, uint256 creditedAmount, uint256 resultingResurrectionPool);

    error ModulusNotConfigured();
    error InvalidDecayBase();

    /// @notice Records a failed execution and triggers the resurrection path.
    function recordFailure(bytes32 execId, uint256 value) public virtual {
        executions[execId] = ExecutionRecord(block.timestamp, value, false);
        emit FailureRecorded(execId, value);
        _resurrect(execId, value);
    }

    function _resurrect(bytes32 execId, uint256 value) internal {
        ExecutionRecord storage rec = executions[execId];
        uint256 deltaT = block.timestamp - rec.timestamp;

        if (modulus == 0) revert ModulusNotConfigured();
        uint256 entropySample = _entropy(execId, value);
        uint256 raw = entropySample % modulus;

        uint256 decay = _decay(deltaT);
        uint256 resurrectable = _mulWadDown(raw, decay);

        uint256 forwardMirror = _forwardLiquidityReference();
        uint256 clamped = _antiDoubleCount(resurrectable, forwardMirror);
        _enforceAntiGaming(clamped, value, forwardMirror);

        resurrectedLiquidity += clamped;
        require(resurrectedLiquidity + forwardMirror <= _globalLiquidityCap(), "RRE: cap exceeded");

        emit LiquidityResurrected(execId, clamped, resurrectedLiquidity);
    }

    function _entropy(bytes32 execId, uint256 val) internal view returns (uint256) {
        return uint256(
            keccak256(abi.encodePacked(blockhash(block.number - 1), msg.sender, gasleft(), execId, val))
        );
    }

    function _decay(uint256 deltaT) internal view returns (uint256) {
        if (deltaT == 0) {
            return WAD;
        }
        if (lambda > WAD) revert InvalidDecayBase();

        uint256 factor = lambda;
        uint256 exponent = deltaT;
        uint256 result = WAD;

        while (exponent != 0) {
            if (exponent & 1 == 1) {
                result = _mulWadDown(result, factor);
            }
            exponent >>= 1;
            if (exponent != 0) {
                factor = _mulWadDown(factor, factor);
            }
        }

        return result;
    }

    function _antiDoubleCount(uint256 Lr, uint256 credited) internal pure returns (uint256) {
        if (Lr <= credited) {
            return 0;
        }
        return Lr - credited;
    }

    function _enforceAntiGaming(uint256 Lr, uint256 originalO, uint256 forwardMirror) internal view {
        require(Lr <= MAX_RESURRECTION, "RRE: exceeds MAX_RESURRECTION");
        require(Lr < forwardMirror + originalO, "RRE: gaming path");
    }

    function _mulWadDown(uint256 a, uint256 b) internal pure returns (uint256) {
        return (a * b) / WAD;
    }

    function _forwardLiquidityReference() internal view virtual returns (uint256);

    function _globalLiquidityCap() internal view virtual returns (uint256);
}
