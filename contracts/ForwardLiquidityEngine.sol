// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title ForwardLiquidityEngine
/// @notice Deterministic forward opportunity accrual (F-Engine)
/// @dev Gamma and beta are expressed as 1e18 fixed-point scalars.
abstract contract ForwardLiquidityEngine {
    uint256 public forwardLiquidity;
    uint256 public mu_t;
    uint256 public sigma_t;
    uint256 public expectedFlow;
    uint256 public L_max;
    uint256 public gamma;
    uint256 public beta;

    uint256 public constant ENTROPY_MOD = 1e18;
    uint256 internal constant WAD = 1e18;

    event ForwardAccrued(address indexed sender, uint256 entropyAdjFlow, uint256 resultingForwardLiquidity);
    event ExpectationUpdated(uint256 mu, uint256 sigma);
    event LmaxUpdated(uint256 observedOpportunity, uint256 resultingCap);

    /// @notice Accrues forward liquidity using entropy-adjusted expectation.
    /// @param sender Address used to derive the entropy index.
    function accrueForwardLiquidity(address sender) external virtual {
        uint256 entropy = _entropyIndex(sender);
        uint256 adj = (expectedFlow * entropy) / ENTROPY_MOD;
        uint256 effectiveFlow = expectedFlow + adj;

        uint256 delta = (gamma * effectiveFlow) / WAD;
        forwardLiquidity += delta;
        require(forwardLiquidity <= L_max, "FLE: global cap exceeded");

        emit ForwardAccrued(sender, effectiveFlow, forwardLiquidity);
    }

    /// @notice Updates the expectation parameters.
    /// @dev The mu parameter is directly used as expected flow; sigma is persisted for observability.
    function updateExpectation(uint256 mu, uint256 sigma) external virtual {
        mu_t = mu;
        sigma_t = sigma;
        expectedFlow = mu;

        emit ExpectationUpdated(mu, sigma);
    }

    /// @notice Updates the rolling maximum cap using beta scaling.
    /// @param maxObservedOpportunity Observed opportunity metric in base units.
    function updateLmax(uint256 maxObservedOpportunity) external virtual {
        uint256 newCap = (beta * maxObservedOpportunity) / WAD;
        L_max = newCap;

        emit LmaxUpdated(maxObservedOpportunity, newCap);
    }

    function _entropyIndex(address sender) internal view returns (uint256) {
        return uint256(
            keccak256(abi.encodePacked(blockhash(block.number - 1), sender))
        ) % ENTROPY_MOD;
    }
}
