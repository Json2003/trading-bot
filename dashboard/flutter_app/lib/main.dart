import 'package:flutter/material.dart';

void main() {
  runApp(const TradingBotApp());
}

class TradingBotApp extends StatelessWidget {
  const TradingBotApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      home: Scaffold(
        body: Center(child: Text('Trading Bot Dashboard')),
      ),
    );
  }
}
