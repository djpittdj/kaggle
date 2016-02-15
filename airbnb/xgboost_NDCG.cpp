/* add this function in xgboost/src/learner/evaluation-inl.hpp */

typedef std::pair<double, int> prob_rank_pair;
bool compare_prob_rank_pair (const prob_rank_pair& l, const prob_rank_pair& r) {
  return (l.first > r.first);
}

/*! \brief airbnb NDCG */
struct EvalAirbnbNDCG : public EvalMClassBase<EvalAirbnbNDCG> {
  virtual const char *Name(void) const {
    return "airbnbNDCG";
  }

  inline static float EvalRow(int label, const float *pred, size_t nclass) {
    std::vector<prob_rank_pair> vec_pairs;
    for (size_t i=0; i<nclass; ++i) {
      vec_pairs.push_back(prob_rank_pair(pred[i], static_cast<int>(i)));
    }
    
    std::sort(vec_pairs.begin(), vec_pairs.end(), compare_prob_rank_pair);

    double dcg = 0.0;
    for (size_t i=0; i<5; ++i) {
      int ii = static_cast<int>(i);
      double rel = (vec_pairs.at(ii).second == label);
      dcg += rel/log2(ii+2.0);
    }
    return dcg;
  }
};
