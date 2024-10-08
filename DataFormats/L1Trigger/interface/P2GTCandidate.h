#ifndef DataFormats_L1Trigger_P2GTCandidate_h
#define DataFormats_L1Trigger_P2GTCandidate_h

#include <vector>
#include <ap_int.h>
#include <stdexcept>

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"

namespace l1t {

  class L1GTProducer;

  class P2GTCandidate;
  typedef std::vector<P2GTCandidate> P2GTCandidateCollection;
  typedef edm::Ref<P2GTCandidateCollection> P2GTCandidateRef;
  typedef edm::RefVector<P2GTCandidateCollection> P2GTCandidateRefVector;
  typedef std::vector<P2GTCandidateRef> P2GTCandidateVectorRef;

  class P2GTCandidate : public reco::LeafCandidate {
  public:
    using Base = reco::LeafCandidate;

    using Base::Base;

    friend class L1GTProducer;

    typedef ap_uint<16> hwPT_t;
    typedef ap_int<13> hwPhi_t;
    typedef ap_int<14> hwEta_t;
    typedef ap_int<18> hwZ0_t;
    typedef ap_uint<11> hwIsolationPT_t;
    typedef ap_uint<6> hwQualityFlags_t;
    typedef ap_uint<10> hwQualityScore_t;
    typedef ap_uint<1> hwCharge_t;
    typedef ap_int<12> hwD0_t;
    typedef ap_uint<4> hwBeta_t;
    typedef ap_uint<10> hwMass_t;
    typedef ap_uint<16> hwIndex_t;
    typedef ap_uint<10> hwSeed_pT_t;
    typedef ap_int<10> hwSeed_z0_t;
    typedef ap_uint<16> hwScalarSumPT_t;
    typedef ap_uint<5> hwNumber_of_tracks_t;
    typedef ap_uint<4> hwNumber_of_displaced_tracks_t;
    typedef ap_uint<12> hwSum_pT_pv_t;
    typedef ap_uint<2> hwType_t;
    typedef ap_uint<8> hwNumber_of_tracks_in_pv_t;
    typedef ap_uint<10> hwNumber_of_tracks_not_in_pv_t;

    // Similar to std::optional<T> but avoids inheritance for ROOT file embedding
    template <typename T>
    struct Optional {
      Optional() : value_(0), set_(false) {}
      Optional(T value) : value_(value), set_(true) {}

      operator T() const { return value_; }
      operator bool() const { return set_; }

      bool operator==(bool rhs) const { return set_ == rhs; }
      bool operator!=(bool rhs) const { return set_ != rhs; }

    private:
      T value_;
      bool set_;
    };

    enum ObjectType {
      Undefined,
      GCTNonIsoEg,
      GCTIsoEg,
      GCTJets,
      GCTTaus,
      GCTHtSum,
      GCTEtSum,
      GMTSaPromptMuons,
      GMTSaDisplacedMuons,
      GMTTkMuons,
      GMTTopo,
      GTTPromptJets,
      GTTDisplacedJets,
      GTTPhiCandidates,
      GTTRhoCandidates,
      GTTBsCandidates,
      GTTHadronicTaus,
      GTTPromptTracks,
      GTTDisplacedTracks,
      GTTPrimaryVert,
      GTTPromptHtSum,
      GTTDisplacedHtSum,
      GTTEtSum,
      CL2JetsSC4,
      CL2JetsSC8,
      CL2Taus,
      CL2Electrons,
      CL2Photons,
      CL2HtSum,
      CL2EtSum
    };

    void setHwPT(hwPT_t hwPT) { hwPT_ = hwPT.to_int(); }
    void setHwPhi(hwPhi_t hwPhi) { hwPhi_ = hwPhi.to_int(); }
    void setHwEta(hwEta_t hwEta) { hwEta_ = hwEta.to_int(); }
    void setHwZ0(hwZ0_t hwZ0) { hwZ0_ = hwZ0.to_int(); }
    void setHwIsolationPT(hwIsolationPT_t hwIso) { hwIsolationPT_ = hwIso.to_int(); }
    void setHwQualityFlags(hwQualityFlags_t hwQualityFlags) { hwQualityFlags_ = hwQualityFlags.to_int(); }
    void setHwQualityScore(hwQualityScore_t hwQualityScore) { hwQualityScore_ = hwQualityScore.to_int(); }
    void setHwCharge(hwCharge_t hwCharge) { hwCharge_ = hwCharge.to_int(); }
    void setHwD0(hwD0_t hwD0) { hwD0_ = hwD0.to_int(); }
    void setHwBeta(hwBeta_t hwBeta) { hwBeta_ = hwBeta.to_int(); }
    void setHwMass(hwMass_t hwMass) { hwMass_ = hwMass.to_int(); }
    void setHwIndex(hwIndex_t hwIndex) { hwIndex_ = hwIndex.to_int(); }
    void setHwSeed_pT(hwSeed_pT_t hwSeed_pT) { hwSeed_pT_ = hwSeed_pT.to_int(); }
    void setHwSeed_z0(hwSeed_z0_t hwSeed_z0) { hwSeed_z0_ = hwSeed_z0.to_int(); }
    void setHwScalarSumPT(hwScalarSumPT_t hwScalarSumPT) { hwScalarSumPT_ = hwScalarSumPT.to_int(); }
    void setHwNumber_of_tracks(hwNumber_of_tracks_t hwNumber_of_tracks) {
      hwNumber_of_tracks_ = hwNumber_of_tracks.to_int();
    }

    void setHwNumber_of_displaced_tracks(hwNumber_of_displaced_tracks_t hwNumber_of_displaced_tracks) {
      hwNumber_of_displaced_tracks_ = hwNumber_of_displaced_tracks.to_int();
    }

    void setHwSum_pT_pv(hwSum_pT_pv_t hwSum_pT_pv) { hwSum_pT_pv_ = hwSum_pT_pv.to_int(); }
    void setHwType(hwType_t hwType) { hwType_ = hwType.to_int(); }
    void setHwNumber_of_tracks_in_pv(hwNumber_of_tracks_in_pv_t hwNumber_of_tracks_in_pv) {
      hwNumber_of_tracks_in_pv_ = hwNumber_of_tracks_in_pv.to_int();
    }
    void setHwNumber_of_tracks_not_in_pv(hwNumber_of_tracks_not_in_pv_t hwNumber_of_tracks_not_in_pv) {
      hwNumber_of_tracks_not_in_pv_ = hwNumber_of_tracks_not_in_pv.to_int();
    }

    hwPT_t hwPT() const {
      if (!hwPT_) {
        throw std::invalid_argument("Object doesn't have pT");
      }
      return static_cast<int>(hwPT_);
    }

    hwPhi_t hwPhi() const {
      if (!hwPhi_) {
        throw std::invalid_argument("Object doesn't have phi");
      }
      return static_cast<int>(hwPhi_);
    }

    hwEta_t hwEta() const {
      if (!hwEta_) {
        throw std::invalid_argument("Object doesn't have eta");
      }
      return static_cast<int>(hwEta_);
    }

    hwZ0_t hwZ0() const {
      if (!hwZ0_) {
        throw std::invalid_argument("Object doesn't have z0");
      }
      return static_cast<int>(hwZ0_);
    }

    hwIsolationPT_t hwIsolationPT() const {
      if (!hwIsolationPT_) {
        throw std::invalid_argument("Object doesn't have isolationPT");
      }
      return static_cast<int>(hwIsolationPT_);
    }

    hwQualityFlags_t hwQualityFlags() const {
      if (!hwQualityFlags_) {
        throw std::invalid_argument("Object doesn't have qualityFlags");
      }
      return static_cast<int>(hwQualityFlags_);
    }

    hwQualityScore_t hwQualityScore() const {
      if (!hwQualityScore_) {
        throw std::invalid_argument("Object doesn't have qualityScore");
      }
      return static_cast<int>(hwQualityScore_);
    }

    hwCharge_t hwCharge() const {
      if (!hwCharge_) {
        throw std::invalid_argument("Object doesn't have charge");
      }
      return static_cast<int>(hwCharge_);
    }

    hwD0_t hwD0() const {
      if (!hwD0_) {
        throw std::invalid_argument("Object doesn't have d0");
      }
      return static_cast<int>(hwD0_);
    }

    hwBeta_t hwBeta() const {
      if (!hwBeta_) {
        throw std::invalid_argument("Object doesn't have beta");
      }
      return static_cast<int>(hwBeta_);
    }

    hwMass_t hwMass() const {
      if (!hwMass_) {
        throw std::invalid_argument("Object doesn't have mass");
      }
      return static_cast<int>(hwMass_);
    }

    hwIndex_t hwIndex() const {
      if (!hwIndex_) {
        throw std::invalid_argument("Object doesn't have index");
      }
      return static_cast<int>(hwIndex_);
    }

    hwSeed_pT_t hwSeed_pT() const {
      if (!hwSeed_pT_) {
        throw std::invalid_argument("Object doesn't have seed_pT");
      }
      return static_cast<int>(hwSeed_pT_);
    }

    hwSeed_z0_t hwSeed_z0() const {
      if (!hwSeed_z0_) {
        throw std::invalid_argument("Object doesn't have seed_z0");
      }
      return static_cast<int>(hwSeed_z0_);
    }

    hwScalarSumPT_t hwScalarSumPT() const {
      if (!hwScalarSumPT_) {
        throw std::invalid_argument("Object doesn't have scalarSumPT");
      }
      return static_cast<int>(hwScalarSumPT_);
    }

    hwNumber_of_tracks_t hwNumber_of_tracks() const {
      if (!hwNumber_of_tracks_) {
        throw std::invalid_argument("Object doesn't have number_of_tracks");
      }
      return static_cast<int>(hwNumber_of_tracks_);
    }

    hwNumber_of_displaced_tracks_t hwNumber_of_displaced_tracks() const {
      if (!hwNumber_of_displaced_tracks_) {
        throw std::invalid_argument("Object doesn't have hwNumber_of_displaced_tracks");
      }
      return static_cast<int>(hwNumber_of_displaced_tracks_);
    }

    hwSum_pT_pv_t hwSum_pT_pv() const {
      if (!hwSum_pT_pv_) {
        throw std::invalid_argument("Object doesn't have sum_pT_pv");
      }
      return static_cast<int>(hwSum_pT_pv_);
    }

    hwType_t hwType() const {
      if (!hwType_) {
        throw std::invalid_argument("Object doesn't have type");
      }
      return static_cast<int>(hwType_);
    }

    hwNumber_of_tracks_in_pv_t hwNumber_of_tracks_in_pv() const {
      if (!hwNumber_of_tracks_in_pv_) {
        throw std::invalid_argument("Object doesn't have number_of_tracks_in_pv");
      }
      return static_cast<int>(hwNumber_of_tracks_in_pv_);
    }

    hwNumber_of_tracks_not_in_pv_t hwNumber_of_tracks_not_in_pv() const {
      if (!hwNumber_of_tracks_not_in_pv_) {
        throw std::invalid_argument("Object doesn't have hwNumber_of_tracks_not_in_pv");
      }
      return static_cast<int>(hwNumber_of_tracks_not_in_pv_);
    }

    ObjectType objectType() const { return objectType_; }

    // Nano SimpleCandidateFlatTableProducer accessor functions
    int hwPT_toInt() const { return hwPT().to_int(); }
    int hwPhi_toInt() const { return hwPhi().to_int(); }
    int hwEta_toInt() const { return hwEta().to_int(); }
    int hwZ0_toInt() const { return hwZ0().to_int(); }
    int hwIsolationPT_toInt() const { return hwIsolationPT().to_int(); }
    int hwQualityFlags_toInt() const { return hwQualityFlags().to_int(); }
    int hwQualityScore_toInt() const { return hwQualityScore().to_int(); }
    int hwCharge_toInt() const { return hwCharge().to_int(); }
    int hwD0_toInt() const { return hwD0().to_int(); }
    int hwBeta_toInt() const { return hwBeta().to_int(); }
    int hwMass_toInt() const { return hwMass().to_int(); }
    int hwIndex_toInt() const { return hwIndex().to_int(); }
    int hwSeed_pT_toInt() const { return hwSeed_pT().to_int(); }
    int hwSeed_z0_toInt() const { return hwSeed_z0().to_int(); }
    int hwScalarSumPT_toInt() const { return hwScalarSumPT().to_int(); }
    int hwNumber_of_tracks_toInt() const { return hwNumber_of_tracks().to_int(); }
    int hwNumber_of_displaced_tracks_toInt() const { return hwNumber_of_displaced_tracks().to_int(); }
    int hwSum_pT_pv_toInt() const { return hwSum_pT_pv().to_int(); }
    int hwType_toInt() const { return hwType().to_int(); }
    int hwNumber_of_tracks_in_pv_toInt() const { return hwNumber_of_tracks_in_pv().to_int(); }
    int hwNumber_of_tracks_not_in_pv_toInt() const { return hwNumber_of_tracks_not_in_pv().to_int(); }

    bool operator==(const P2GTCandidate& rhs) const;
    bool operator!=(const P2GTCandidate& rhs) const;

    bool isElectron() const override { return objectType_ == CL2Electrons; };

    bool isMuon() const override {
      return objectType_ == GMTSaPromptMuons || objectType_ == GMTSaDisplacedMuons || objectType_ == GMTTkMuons;
    };

    bool isStandAloneMuon() const override {
      return objectType_ == GMTSaPromptMuons || objectType_ == GMTSaDisplacedMuons;
    };

    bool isTrackerMuon() const override { return objectType_ == GMTTkMuons; }

    bool isPhoton() const override { return objectType_ == CL2Photons; }

    bool isJet() const override {
      return objectType_ == GCTJets || objectType_ == GTTPromptJets || objectType_ == GTTDisplacedJets ||
             objectType_ == CL2JetsSC4 || objectType_ == CL2JetsSC8;
    }

  private:
    Optional<int> hwPT_;
    Optional<int> hwPhi_;
    Optional<int> hwEta_;
    Optional<int> hwZ0_;
    Optional<int> hwIsolationPT_;
    Optional<int> hwQualityFlags_;
    Optional<int> hwQualityScore_;
    Optional<int> hwCharge_;
    Optional<int> hwD0_;
    Optional<int> hwBeta_;
    Optional<int> hwMass_;
    Optional<int> hwIndex_;
    Optional<int> hwSeed_pT_;
    Optional<int> hwSeed_z0_;
    Optional<int> hwScalarSumPT_;
    Optional<int> hwNumber_of_tracks_;
    Optional<int> hwNumber_of_displaced_tracks_;

    // TODO ?
    Optional<int> hwSum_pT_pv_;
    Optional<int> hwType_;
    Optional<int> hwNumber_of_tracks_in_pv_;
    Optional<int> hwNumber_of_tracks_not_in_pv_;

    ObjectType objectType_ = Undefined;
  };

};  // namespace l1t

#endif  // DataFormats_L1Trigger_P2GTCandidate_h
