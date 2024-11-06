//== MutableMemberAccessChecker.cpp - Checks for accessing mutable members via const pointer --------------*- C++ -*--==//
//
// By Ivan Razumov <ivan.razumov@cern.ch>
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
  void MutableMemberAccessChecker::checkPreStmt(const clang::MemberExpr *ME, clang::ento::CheckerContext &C) const {
    // Common checks

    // == Filter out classes with "safe" names ==
    const auto *RD = llvm::dyn_cast<clang::CXXRecordDecl>(ME->getMemberDecl()->getDeclContext());
    if (RD) {
      std::string ClassName = RD->getNameAsString();
      if (support::isSafeClassName(ClassName)) {
        return;  // Skip checking for this class
      }
    }

    // == Check attributes ==
    const clang::FunctionDecl *FuncD = C.getLocationContext()->getStackFrame()->getDecl()->getAsFunction();
    const clang::AttrVec &Attrs = FuncD->getAttrs();
    for (const auto *A : Attrs) {
      if (clang::isa<clang::CMSThreadGuardAttr>(A) || clang::isa<clang::CMSThreadSafeAttr>(A) ||
          clang::isa<clang::CMSSaAllowAttr>(A)) {
        return;  // Attribute found, do not emit an error.
      }
    }

    // == Check if this is a cmssw local file ==
    // Create a PathDiagnosticLocation for reporting
    clang::ento::PathDiagnosticLocation PathLoc =
        clang::ento::PathDiagnosticLocation::createBegin(ME, C.getSourceManager(), C.getLocationContext());

    // Get the BugReporter instance from the CheckerContext
    clang::ento::BugReporter &BR = C.getBugReporter();

    if (!m_exception.reportMutableMember(PathLoc, BR)) {
      return;
    }

    // == Only proceed if the member is mutable ==
    const auto *FD = llvm::dyn_cast<clang::FieldDecl>(ME->getMemberDecl());
    if (!FD || !FD->isMutable()) {
      return;  // Skip if it's not a mutable field
    }

    // == Check if we are inside a const-qualified member function ==
    bool isInConstMemberFunc = false;
    const auto *MethodDecl = llvm::dyn_cast<clang::CXXMethodDecl>(FuncD);
    if (MethodDecl && MethodDecl->isConst()) {
      isInConstMemberFunc = true;
    }

    if (!isInConstMemberFunc) {
      return;
    }

    bool ret;
    ret = checkAssignToMutable(ME, C, FuncD);
    if (!ret)
      ret = checkCallNonConstOfMutable(ME, C);

    if (ret) {
      if (RD) {
        std::string ClassName = RD->getNameAsString();
        std::string MemberName = ME->getMemberDecl()->getNameAsString();
        std::string FunctionName = MethodDecl->getNameAsString();
        std::string tname = "mutablemember-checker.txt.unsorted";
        std::string ostring = "flagged class '" + ClassName + "' modifying mutable member '" + MemberName +
                              "' in function '" + FunctionName + "'";
        support::writeLog(ostring, tname);
      }
    }
  }  // checkPreStmt

  // Check direct modifications of mutable (assign, compound stmt, increment/decrement)
  bool MutableMemberAccessChecker::checkAssignToMutable(const clang::MemberExpr *ME,
                                                        clang::ento::CheckerContext &C,
                                                        const clang::FunctionDecl *FuncD) const {
    // == Check if this is a modifying statement ==
    bool isModification = false;

    // Retrieve the parent statement of the MemberExpr
    const clang::LocationContext *LC = C.getLocationContext();
    const clang::ParentMap &PM = LC->getParentMap();
    const clang::Stmt *ParentStmt = PM.getParent(ME);

    if (!ParentStmt) {
      return false;
    }

    // Check if it is an assignment operator (binary operator)
    if (const auto *BO = llvm::dyn_cast<clang::BinaryOperator>(ParentStmt)) {
      if (BO->isAssignmentOp() && BO->getLHS() == ME) {
        // The MemberExpr is on the left-hand side of an assignment
        isModification = true;
      }
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
    if (!BT) {
      BT = std::make_unique<clang::ento::BugType>(
          this, "Mutable member modification in const member function", "ConstThreadSafety");
    }
    auto Report = std::make_unique<clang::ento::PathSensitiveBugReport>(
        *BT, "Modifying mutable member in const member function is potentially thread-unsafe", C.generateErrorNode());
    Report->addRange(ME->getSourceRange());
    C.emitReport(std::move(Report));

    return true;
  }  // checkAssignToMutable

  // Check for indirect modifications of mutable (calling non-const method)
  bool MutableMemberAccessChecker::checkCallNonConstOfMutable(const clang::MemberExpr *ME,
                                                              clang::ento::CheckerContext &C) const {
    // Traverse upwards to check if the MemberExpr is part of a CXXMemberCallExpr
    const clang::Expr *E = ME;
    while (E) {
      if (const clang::CXXMemberCallExpr *Call = llvm::dyn_cast<clang::CXXMemberCallExpr>(E->IgnoreParenCasts())) {
        const clang::CXXMethodDecl *CalledMethod = Call->getMethodDecl();
        if (CalledMethod && !CalledMethod->isConst()) {
          // Report an issue
          if (!BT) {
            BT = std::make_unique<clang::ento::BugType>(
                this, "Mutable member modification in const member function", "ConstThreadSafety");
          }
          auto Report = std::make_unique<clang::ento::PathSensitiveBugReport>(
              *BT,
              "Modifying mutable member in const member function is potentially thread-unsafe",
              C.generateErrorNode());
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
  }
}  // namespace clangcms
