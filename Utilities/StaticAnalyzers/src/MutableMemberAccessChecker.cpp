//== MutableMemberChecker.cpp - Checks for accessing mutable members via const pointer --------------*- C++ -*--==//
//
// By Thomas Hauth [ Thomas.Hauth@cern.ch ], updated by Ivan Razumov <ivan.razumov@cern.ch>
//
//===----------------------------------------------------------------------===//

#include "MutableMemberAccessChecker.h"
#include <clang/AST/Stmt.h>
#include <clang/AST/Expr.h>
#include <clang/AST/ExprCXX.h>
#include <clang/AST/ParentMap.h>
#include <clang/Analysis/AnalysisDeclContext.h>
#include <clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h>

namespace clangcms {
/*
  void printLocation(const clang::MemberExpr *ME, clang::ento::CheckerContext &C) {
    clang::SourceLocation Loc = ME->getExprLoc();
    const clang::SourceManager &SM = C.getSourceManager();
    clang::PresumedLoc PLoc = SM.getPresumedLoc(Loc);

    if (PLoc.isValid()) {
      llvm::StringRef FileName = PLoc.getFilename();
      unsigned LineNumber = PLoc.getLine();
      unsigned ColumnNumber = PLoc.getColumn();

      // Print or log the information
      llvm::errs() << "File: " << FileName << ", Line: " << LineNumber << ", Column: " << ColumnNumber << "\n";
    } else {
      llvm::errs() << "Invalid source location\n";
    }
  }
*/
  // void printLocation(const clang::FieldDecl *D, clang::ento::AnalysisManager &Mgr) {
  //   // Obtain the source location of the declaration
  //   clang::SourceLocation Loc = D->getLocation();

  //   // Access the SourceManager from the AnalysisManager
  //   const clang::SourceManager &SM = Mgr.getSourceManager();

  //   // Get presumed location (file name, line, column)
  //   clang::PresumedLoc PLoc = SM.getPresumedLoc(Loc);

  //   if (PLoc.isValid()) {
  //     llvm::StringRef FileName = PLoc.getFilename();
  //     unsigned LineNumber = PLoc.getLine();
  //     unsigned ColumnNumber = PLoc.getColumn();

  //     // Print or log the information
  //     llvm::errs() << "FieldDecl File: " << FileName << ", Line: " << LineNumber << ", Column: " << ColumnNumber
  //                  << "\n";
  //   } else {
  //     llvm::errs() << "Invalid source location for FieldDecl\n";
  //   }
  // }

  void MutableMemberChecker::checkPreStmt(const clang::MemberExpr *ME, clang::ento::CheckerContext &C) const {
    // Common checks
    // printLocation(ME, C);

    // == Filter out classes with "safe" names ==
    const auto *RD = llvm::dyn_cast<clang::CXXRecordDecl>(ME->getMemberDecl()->getDeclContext());
    if (RD) {
      std::string ClassName = RD->getNameAsString();
      if (support::isSafeClassName(ClassName)) {
        // llvm::errs() << "Skip: class " << ClassName << " is safe\n";
        return;  // Skip checking for this class
      }
    }

    // == Check attributes ==
    const clang::FunctionDecl *FuncD = C.getLocationContext()->getStackFrame()->getDecl()->getAsFunction();
    const clang::AttrVec &Attrs = FuncD->getAttrs();
    for (const auto *A : Attrs) {
      // // llvm::errs() << "FunctionDecl Attribute " << A->getNormalizedFullName () << "\n";
      if (clang::isa<clang::CMSThreadGuardAttr>(A) || clang::isa<clang::CMSThreadSafeAttr>(A) ||
          clang::isa<clang::CMSSaAllowAttr>(A)) {
        // llvm::errs() << "Skip: function decorated with cms attribute\n";
        return;
      }
    }

    // == Check if this is a cmssw local file ==
    // Create a PathDiagnosticLocation for reporting
    clang::ento::PathDiagnosticLocation PathLoc =
        clang::ento::PathDiagnosticLocation::createBegin(ME, C.getSourceManager(), C.getLocationContext());

    // Get the BugReporter instance from the CheckerContext
    clang::ento::BugReporter &BR = C.getBugReporter();

    if (!m_exception.reportMutableMember(PathLoc, BR)) {
      // // llvm::errs() << "Skip: non-local file\n";
      return;
    }

    // == Only proceed if the member is mutable ==
    const auto *FD = llvm::dyn_cast<clang::FieldDecl>(ME->getMemberDecl());
    if (!FD || !FD->isMutable()) {
      // llvm::errs() << "Skip: not mutable\n";
      return;  // Skip if it's not a mutable field
    }
    // llvm::errs() << "Processing field " << FD->getNameAsString() << "\n";
    const clang::AttrVec &FAttrs = FD->getAttrs();
    // llvm::errs() << "FD has " << FAttrs.size() << " attrs\n";
    for (const auto *A : FAttrs) {
      if (clang::isa<clang::CMSThreadGuardAttr>(A) || clang::isa<clang::CMSThreadSafeAttr>(A) ||
          clang::isa<clang::CMSSaAllowAttr>(A)) {
        // llvm::errs() << "Skip: member decorated with cms attribute\n";
        return;
      }
    }
    // llvm::errs() << "FieldDecl not decorated\n";

    // == Check if we are inside a const-qualified member function ==
    const auto *MethodDecl = llvm::dyn_cast<clang::CXXMethodDecl>(FuncD);
    if (!MethodDecl || !MethodDecl->isConst()) {
      // llvm::errs() << "Skip: not method or non const method\n";
      return;
    }

    bool ret;
    ret = checkAssignToMutable(ME, C, FuncD);
    // llvm::errs() << "checkAssignToMutable returned " << (ret ? "yeah\n" : "nope\n");
    if (!ret) {
      ret = checkCallNonConstOfMutable(ME, C);
      // llvm::errs() << "checkCallNonConstOfMutable returned " << (ret ? "yeah\n" : "nope\n");
    }

    if (ret) {
      // // llvm::errs() << "will save mutable member " << FD->getNameAsString() << "\n";
      if (RD) {
        std::string ClassName = RD->getNameAsString();
        std::string MemberName = ME->getMemberDecl()->getNameAsString();
        // // llvm::errs() << "do save mutable member " << ClassName << "::" << MemberName << "\n";
        std::string FunctionName = MethodDecl->getNameAsString();
        std::string tname = "mutablemember-checker.txt.unsorted";
        std::string ostring = "flagged class '" + ClassName + "' modifying mutable member '" + MemberName +
                              "' in function '" + FunctionName + "'";
        support::writeLog(ostring, tname);
        // ModifiedMutableMembers.insert(FD);
      }
    }
  }  // checkPreStmt

  // Check direct modifications of mutable (assign, compound stmt, increment/decrement)
  bool MutableMemberChecker::checkAssignToMutable(const clang::MemberExpr *ME,
                                                  clang::ento::CheckerContext &C,
                                                  const clang::FunctionDecl *FuncD) const {
    // == Check if this is a modifying statement ==
    bool isModification = false;

    // Retrieve the parent statement of the MemberExpr
    const clang::LocationContext *LC = C.getLocationContext();
    const clang::ParentMap &PM = LC->getParentMap();
    const clang::Stmt *ParentStmt = PM.getParent(ME);

    if (!ParentStmt) {
      // llvm::errs() << "No parent stmt\n";
      return false;
    }

    // Check if it is an assignment operator (binary operator)
    const auto *BO = llvm::dyn_cast<clang::BinaryOperator>(ParentStmt);
    if (BO) {
      // llvm::errs() << "isAssignmentOp -> " << (BO->isAssignmentOp() ? "yeah" : "nope") << "\n";
      const auto *LHSAsME = llvm::dyn_cast<clang::MemberExpr>(BO->getLHS());
      if (LHSAsME) {
        if (BO->isAssignmentOp() && LHSAsME == ME) {
          // The MemberExpr is on the left-hand side of an assignment
          // llvm::errs() << "LHSasME is ME\n";
          isModification = true;
        } else {
          // llvm::errs() << "LHSasME is not ME\n";
        }
      } else {
        // llvm::errs() << "LHS is not MemberExpr\n";
      }
    } else {
      // llvm::errs() << "Not a binary op\n";
      // ParentStmt->dump();
    }

    // Check if it is an overloaded assignment operator
    const auto *CO = llvm::dyn_cast<clang::CXXOperatorCallExpr>(ParentStmt);
    if (CO) {
      // llvm::errs() << "isAssignmentOp2 -> " << (CO->isAssignmentOp() ? "yeah" : "nope") << "\n";
      const auto *LHSAsME = llvm::dyn_cast<clang::MemberExpr>(CO->getArg(0));
      if (LHSAsME) {
        if (CO->isAssignmentOp() && LHSAsME == ME) {
          // llvm::errs() << "LHSasME is ME\n";
          // The MemberExpr is on the left-hand side of an assignment
          isModification = true;
        } else {
          // llvm::errs() << "LHSasME is not ME\n";
        }
      } else {
        // llvm::errs() << "LHS is not MemberExpr\n";
      }
    } else {
      // llvm::errs() << "Not a CXXOperatorCall op\n";
      // ParentStmt->dump();
    }

    // Check for increment/decrement
    if (const auto *UO = llvm::dyn_cast<clang::UnaryOperator>(ParentStmt)) {
      if (UO->isIncrementDecrementOp() && UO->getSubExpr() == ME) {
        isModification = true;
      }
    }

    if (!isModification) {
      return false;
    }

    // == Report a bug if none of the above conditions allow access. ==
    std::string MutableMemberName = ME->getMemberDecl()->getQualifiedNameAsString();
    if (!BT) {
      BT = std::make_unique<clang::ento::BugType>(
          this, "Mutable member modification in const member function", "ConstThreadSafety");
    }
    std::string Description =
        "Modifying mutable member '" + MutableMemberName + "' in const member function is potentially thread-unsafe ";
    auto Report = std::make_unique<clang::ento::PathSensitiveBugReport>(*BT, Description, C.generateErrorNode());
    Report->addRange(ME->getSourceRange());
    C.emitReport(std::move(Report));
    return true;
  }  // checkAssignToMutable

  // Check for indirect modifications of mutable (calling non-const method)
  bool MutableMemberChecker::checkCallNonConstOfMutable(const clang::MemberExpr *ME,
                                                        clang::ento::CheckerContext &C) const {
    // Traverse upwards to check if the MemberExpr is part of a CXXMemberCallExpr
    const clang::Expr *E = ME;
    while (E) {
      if (const clang::CXXMemberCallExpr *Call = llvm::dyn_cast<clang::CXXMemberCallExpr>(E->IgnoreParenCasts())) {
        const clang::CXXMethodDecl *CalledMethod = Call->getMethodDecl();
        if (CalledMethod && !CalledMethod->isConst()) {
          // Get the name of the mutable member
          std::string MutableMemberName = ME->getMemberDecl()->getQualifiedNameAsString();

          // Get the name of the called method
          std::string CalledMethodName = CalledMethod->getQualifiedNameAsString();
          // Report an issue
          if (!BT) {
            BT = std::make_unique<clang::ento::BugType>(
                this, "Mutable member modification in const member function", "ConstThreadSafety");
          }
          std::string Description = "Calling non-const method '" + CalledMethodName + "' of mutable member '" +
                                    MutableMemberName + "' in a const member function is potentially thread-unsafe.";
          auto Report = std::make_unique<clang::ento::PathSensitiveBugReport>(*BT, Description, C.generateErrorNode());
          Report->addRange(ME->getSourceRange());
          C.emitReport(std::move(Report));
          return true;
        }
      }
      // Move up to the parent expression
      const clang::Stmt *ParentStmt = C.getLocationContext()->getParentMap().getParent(E);
      E = llvm::dyn_cast_or_null<clang::Expr>(ParentStmt);
    }
    return false;
  }  // checkCallNonConstOfMutable

  // void MutableMemberChecker::checkASTDecl(const clang::FieldDecl *D,
  //                                         clang::ento::AnalysisManager &Mgr,
  //                                         clang::ento::BugReporter &BR) const {
  //   if (D->isMutable()) {
  //     printLocation(D, Mgr);
  //     // // llvm::errs() << "Mutable field found: " << D->getNameAsString() << "\n";

  //     // Retrieve the source location of the mutable field
  //     clang::SourceLocation Loc = D->getLocation();

  //     // Create a PathDiagnosticLocation for the BugReporter
  //     clang::ento::PathDiagnosticLocation PathLoc =
  //         clang::ento::PathDiagnosticLocation::createBegin(D, Mgr.getSourceManager());

  //     // Call CmsException::reportMutableMember
  //     if (!m_exception.reportMutableMember(PathLoc, BR)) {
  //       // // llvm::errs() << "skip non local\n";
  //       return;
  //     }

  //     MutableMembers.insert(D);
  //   }
  // }  // checkASTDecl

  // void MutableMemberChecker::checkEndAnalysis(clang::ento::ExplodedGraph &G,
  //                                             clang::ento::BugReporter &BR,
  //                                             clang::ento::ExprEngine &Eng) const {
  //   // llvm::errs() << "MutableMembers:\n";
  //   for (const auto *Field : MutableMembers) {
  //     // llvm::errs() << "\t" << Field->getNameAsString() << "\n";
  //   }
  //   // llvm::errs() << "ModifiedMutableMembers:\n";
  //   for (const auto *Field : ModifiedMutableMembers) {
  //     // llvm::errs() << "\t" << Field->getNameAsString() << "\n";
  //   }

  //   for (const auto *Field : MutableMembers) {
  //     if (!ModifiedMutableMembers.count(Field)) {
  //       reportUselessMutableField(Field, BR);
  //     }
  //   }
  // }

  // void MutableMemberChecker::reportUselessMutableField(const clang::FieldDecl *Field,
  //                                                      clang::ento::BugReporter &BR) const {
  //   // Create a location for the diagnostic based on where the field is declared
  //   clang::ento::PathDiagnosticLocation DLoc =
  //       clang::ento::PathDiagnosticLocation::createBegin(Field, BR.getSourceManager());

  //   // Emit a basic report with a message, using the field's name and location
  //   BR.EmitBasicReport(Field,
  //                      this,
  //                      "Useless mutable field",
  //                      "ConstThreadSafety",
  //                      "The mutable field '" + Field->getNameAsString() + "' is not modified in any const methods",
  //                      DLoc);
  // }  //reportUselessMutableField

}  // namespace clangcms
