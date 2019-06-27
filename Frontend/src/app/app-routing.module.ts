import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';
import { HomeComponent } from './home/home.component';
import { PredictionsComponent } from './predictions/predictions.component';
import { NotSupportComponent } from './not-support/not-support.component';

const routes: Routes = [
  { path: 'home', component: HomeComponent },
  { path: 'tagging', component: NotSupportComponent },
  { path: 'predictions', component: PredictionsComponent },
  { path: 'performances', component: NotSupportComponent },
  { path: '',   redirectTo: '/home', pathMatch: 'full' },
  { path: '**',   redirectTo: '/home', pathMatch: 'full' }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }

