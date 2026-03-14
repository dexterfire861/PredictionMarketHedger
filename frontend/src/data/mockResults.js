export const MOCK_RESULTS = {
  risks: [
    {
      event: 'China tariff increase to 60%+',
      description:
        'US imposes additional tariffs on Chinese imports, pushing effective rate above 60%. Would directly increase COGS by 25-40% on affected components.',
      exposure_amount: 640000,
      exposure_direction: 'negative',
      timeframe: '6 months',
      confidence: 0.85,
      risk_category: 'tariff',
    },
    {
      event: 'Trade war escalation & retaliatory tariffs',
      description:
        'China retaliates with export controls or tariffs on US goods, causing broader supply chain disruption and component shortages.',
      exposure_amount: 320000,
      exposure_direction: 'negative',
      timeframe: '12 months',
      confidence: 0.65,
      risk_category: 'geopolitical',
    },
    {
      event: 'USD/CNY currency volatility',
      description:
        'Significant yuan depreciation or appreciation affecting import costs independent of tariff changes.',
      exposure_amount: 160000,
      exposure_direction: 'negative',
      timeframe: '6 months',
      confidence: 0.5,
      risk_category: 'currency',
    },
    {
      event: 'Supply chain disruption (Taiwan Strait)',
      description:
        'Military tensions or blockade in Taiwan Strait disrupting semiconductor and electronics supply chains from the region.',
      exposure_amount: 800000,
      exposure_direction: 'negative',
      timeframe: '12 months',
      confidence: 0.3,
      risk_category: 'geopolitical',
    },
  ],
  positions: [
    {
      contract_title: 'U.S. tariff rate on China 15-25% on March 31?',
      contract_source: 'Polymarket',
      side: 'YES',
      current_price: 0.11,
      allocation: 4000,
      url: 'https://polymarket.com',
      ticker: null,
      end_date: '2026-03-31',
      correlation: 'STRONG',
      reasoning:
        'Direct tariff hedge. If China tariffs spike to the 15-25% bracket, this pays out at 9:1. Best risk/reward ratio in the available contract set. Even partial tariff increase triggers full payout.',
      risk: {
        event: 'China tariff increase to 60%+',
      },
    },
    {
      contract_title: 'Will the Court Force Trump to Refund Tariffs?',
      contract_source: 'Polymarket',
      side: 'YES',
      current_price: 0.32,
      allocation: 2500,
      url: 'https://polymarket.com',
      ticker: null,
      end_date: '2026-12-31',
      correlation: 'MODERATE',
      reasoning:
        'Court intervention on tariffs signals policy chaos and potential rollback. Most liquid tariff-adjacent contract at $287K volume. If courts intervene, tariff landscape becomes unpredictable — this position profits from that uncertainty.',
      risk: {
        event: 'Trade war escalation & retaliatory tariffs',
      },
    },
    {
      contract_title: 'Will Trump visit China by April 30?',
      contract_source: 'Polymarket',
      side: 'NO',
      current_price: 0.12,
      allocation: 2000,
      url: 'https://polymarket.com',
      ticker: null,
      end_date: '2026-04-30',
      correlation: 'MODERATE',
      reasoning:
        'Inverse diplomatic hedge. If Trump does NOT visit China, it signals continued tensions and elevated tariff risk. NO at $0.12 gives 8:1 payout. Cheap canary position for diplomatic deterioration.',
      risk: {
        event: 'Trade war escalation & retaliatory tariffs',
      },
    },
    {
      contract_title: '100% tariff on Canada in effect by June 30?',
      contract_source: 'Polymarket',
      side: 'YES',
      current_price: 0.08,
      allocation: 1500,
      url: 'https://polymarket.com',
      ticker: 'KXTARIFFCANADA-26JUN30',
      end_date: '2026-06-30',
      correlation: 'WEAK',
      reasoning:
        'Proxy hedge — if Canada gets hit with 100% tariffs, it signals an extremely aggressive trade posture. China tariffs almost certainly escalate in that scenario. At $0.08 this is a 12:1 payout. Small canary position.',
      risk: {
        event: 'China tariff increase to 60%+',
      },
    },
  ],
  total_cost: 10000,
  total_max_profit: 43695,
  total_exposure: 1920000,
  avg_coverage: 0.186,
  budget_rationale: 'Based on $1.92M total identified exposure and 12% margins, we recommend allocating $10,000 (0.5% of exposure) to cover up to $43,695 in downside scenarios.',
  warnings: [
    'March 31 contracts expire in 17 days — poor duration match for ongoing 6-month business risk. Look for longer-dated tariff contracts when available.',
    'The Canada 100% tariff contract has only $34K total volume. Your $1,500 position represents ~4.4% of the market. Expect slippage on entry.',
    'These positions hedge tariff event risk only. Currency risk (USD/CNY) is not directly hedgeable via current prediction market contracts.',
  ],
  unhedgeable_risks: [
    {
      event: 'USD/CNY currency volatility',
      reason:
        'No prediction market contracts currently cover currency movements. Consider traditional FX hedging instruments (forwards, options) through your bank.',
    },
    {
      event: 'Taiwan Strait supply chain disruption',
      reason:
        'While geopolitical contracts exist, the specific supply chain impact is too indirect. The correlation between military events and your component availability is uncertain.',
    },
  ],
}
